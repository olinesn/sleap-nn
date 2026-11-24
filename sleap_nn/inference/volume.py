"""Inference with the volumetric 3D model (``centered_volume``).

``predict_volume3d`` runs a trained volume on the recording sessions of a ``sio.Labels``.
Per frame:

1. **2D.** Instance groups already on the session (LUC3D's Track All, proofread files)
   are used as they are. Where there are none and centroid + centered-instance models
   are given, each camera's video is run through them.
2. **Matching across cameras** (only for frames without instance groups): the study's
   rule (:func:`sleap_nn.data.multiview.associate_frame`).
3. **Cube centre:** the robust DLT of the volume's ``center_nodes`` (else the mean of
   the triangulated keypoints).
4. **Volume passes:** pass 1, then (by default when the cube centre is not the 2D
   model's anchor) pass 2 re-centred on pass 1's ``center_nodes``.
5. **Outputs:** a ``sio.PredictedInstance3D`` on every instance group (points, and the
   peak voxel probability as point scores), and per session the arrays of LUC3D's
   ``.vol3d.h5`` with ``cov_scale`` measured on the run (as LUC3D's exporter does).

The 3D network runs in float16 on CUDA and reads out in float32 (closer to float32
than the study's bfloat16, and faster); animals go through in fixed-size batches, the
last one padded, because on AMD GPUs every new batch shape costs a MIOpen search.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import attrs
import numpy as np
import sleap_io as sio
import torch
from loguru import logger
from omegaconf import OmegaConf

from sleap_nn.data import multiview as MV
from sleap_nn.data.volume_input import (
    CenteredVolumeDataset,
    FrozenCenteredInstance,
    VolumeInput,
    VolumeSession,
    collate_volume,
    file_sha256,
)
from sleap_nn.io.volume3d import (
    load_chains,
    measure_cov_scale,
    to_rows,
    write_volume3d,
)
from sleap_nn.training.volume import (
    centers_from_points,
    default_passes,
    load_volume_model,
    multiview_config,
    run_volume,
    volume_head,
)


@attrs.define
class SessionVolume:
    """One session's volume output.

    Attributes:
        session_index: Index in ``labels.sessions``.
        node_names: ``J`` part names.
        frames: ``(F,)`` frame indices.
        means: ``(F, A, J, 3)`` 3D points (NaN where no animal).
        covs: ``(F, A, J, 3, 3)`` raw covariances.
        peak: ``(F, A, J)`` peak voxel probability.
        cov_scale: The fusion scale written to the file.
        cov_scale_source: ``"measured"``, ``"validation"`` or ``"given"``.
        cov_scale_points: Points it was measured on.
    """

    session_index: int
    node_names: List[str]
    frames: np.ndarray
    means: np.ndarray
    covs: np.ndarray
    peak: np.ndarray
    cov_scale: Optional[float]
    cov_scale_source: str
    cov_scale_points: int = 0


@attrs.define
class VolumePrediction:
    """What :func:`predict_volume3d` returns: the labels and each session's arrays."""

    labels: sio.Labels
    sessions: List[SessionVolume]


def _device(device: str) -> torch.device:
    if device in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _session_frames(session: sio.RecordingSession) -> List[int]:
    """The frames every camera's video holds, sorted.

    A plain video holds ``range(len(video))``. An HDF5 video with a
    ``frame_numbers`` table (extracted frames, as the 2026-11 study stored them)
    holds its frame numbers, which can be sparse and run past ``len(video)``
    (:func:`sleap_nn.data.providers.video_frame_numbers`).
    """
    from sleap_nn.data.providers import video_frame_numbers

    held = None
    for cam in session.camera_group.cameras:
        video = session.get_video(cam)
        if video is None:
            continue
        numbers = video_frame_numbers(video)
        frames = set(range(len(video))) if numbers is None else set(numbers)
        held = frames if held is None else held & frames
    return sorted(held or ())


def _frames_without_groups(
    session: sio.RecordingSession, frames: Optional[Sequence[int]]
) -> List[int]:
    have = {f for f, fg in session.frame_groups.items() if fg.instance_groups}
    if frames is None:
        frames = _session_frames(session)
    return [int(f) for f in frames if int(f) not in have]


def _run_2d(
    labels: sio.Labels,
    session: sio.RecordingSession,
    frames: List[int],
    model_paths: List[str],
    part_names: Sequence[str],
    max_animals: Optional[int],
    device: str,
    batch_size: int,
):
    """Top-down 2D on each camera for ``frames``, then match cameras into groups."""
    from sleap_nn.inference.run import predict

    cams = list(session.camera_group.cameras)
    arrays = MV.CameraArrays.from_cameras(cams)
    per_cam: Dict[int, Dict[int, List[sio.PredictedInstance]]] = {}
    lf_by: Dict[tuple, sio.LabeledFrame] = {}
    for v, cam in enumerate(cams):
        video = session.get_video(cam)
        if video is None:
            continue
        pred = predict(
            video,
            model_paths=list(model_paths),
            frames=list(frames),
            device=device,
            batch_size=batch_size,
            max_instances=max_animals,
        )
        per_cam[v] = {}
        for lf in pred.labeled_frames:
            insts = [i for i in lf.instances if isinstance(i, sio.PredictedInstance)]
            if not insts:
                continue
            new_lf = sio.LabeledFrame(
                video=video, frame_idx=lf.frame_idx, instances=insts
            )
            labels.append(new_lf)
            lf_by[(v, int(lf.frame_idx))] = new_lf
            per_cam[v][int(lf.frame_idx)] = insts
    n_groups = 0
    for f in frames:
        inst_by_view, objs = {}, {}
        for v, d in per_cam.items():
            insts = d.get(int(f))
            if not insts:
                continue
            objs[v] = insts
            inst_by_view[v] = np.stack(
                [MV._instance_points(i, part_names) for i in insts]
            )
        if len(inst_by_view) < MV.MIN_CAMS:
            continue
        n_max = max_animals or max(len(x) for x in inst_by_view.values())
        animals = MV.associate_frame(arrays, inst_by_view, n_max)
        if not animals:
            continue
        groups = [
            sio.InstanceGroup(
                instance_by_camera={cams[v]: objs[v][i] for v, i in an.items()}
            )
            for an in animals
            if len(an) >= 2
        ]
        if not groups:
            continue
        lfs = {cams[v]: lf_by[(v, int(f))] for v in objs if (v, int(f)) in lf_by}
        session.frame_groups[int(f)] = sio.FrameGroup(
            frame_idx=int(f), instance_groups=groups, labeled_frame_by_camera=lfs
        )
        n_groups += len(groups)
    logger.info(
        f"2D on {len(per_cam)} cameras x {len(frames)} frames; {n_groups} animals "
        "matched across cameras"
    )


def _set_instance_3d(ig: sio.InstanceGroup, points, scores, part_names) -> bool:
    """Put the volume's 3D on a group (keeps a user-placed Instance3D)."""
    cur = ig.instance_3d
    if cur is not None and not isinstance(cur, sio.PredictedInstance3D):
        return False
    skel = next((i.skeleton for i in ig.instances), None)
    if skel is None:
        return False
    J = len(skel.nodes)
    pts = np.full((J, 3), np.nan)
    sc = np.full(J, np.nan)
    for k, n in enumerate(part_names):
        if n in skel.node_names:
            j = skel.node_names.index(n)
            pts[j] = points[k]
            sc[j] = scores[k]
    ig._instance_3d = sio.PredictedInstance3D(
        points=pts,
        skeleton=skel,
        score=float(np.nanmean(sc)) if np.isfinite(sc).any() else None,
        point_scores=sc,
    )
    return True


def predict_volume3d(
    labels: sio.Labels,
    volume_model: str,
    centroid_model: Optional[str] = None,
    centered_instance_model: Optional[str] = None,
    frames: Optional[Sequence[int]] = None,
    passes: Union[str, int] = "auto",
    max_animals: Optional[int] = None,
    allow_other_ci: bool = False,
    cov_scale: Optional[float] = None,
    device: str = "auto",
    batch_size: int = 16,
    net_dtype: Optional[torch.dtype] = torch.float16,
) -> VolumePrediction:
    """Run a trained volume on every recording session of ``labels`` (in place).

    Args:
        labels: Labels with at least one recording session (calibrated cameras and
            their videos). Instance groups already there are used as they are.
        volume_model: The ``centered_volume`` model folder.
        centroid_model: A centroid model folder, for frames without instance groups.
        centered_instance_model: The centered-instance model folder. Default: the one
            the volume was trained on. Another one is refused unless
            ``allow_other_ci`` (a retrained 2D model of the same rig).
        frames: Frames to process (default: every frame with instance groups, plus
            every frame all cameras' videos hold when 2D models are given; for an
            HDF5 video with a ``frame_numbers`` table, its frame numbers).
        passes: ``"auto"`` (2 when the cube centre is not the 2D model's anchor, else
            1), 1 or 2.
        max_animals: Most animals per frame when matching across cameras.
        allow_other_ci: Accept a centered-instance model other than the training one.
        cov_scale: Use this ``cov_scale`` instead of measuring it.
        device: ``"auto"``, ``"cpu"``, ``"cuda"``, ...
        batch_size: Animals per batch (fixed; the last batch is padded).
        net_dtype: Autocast dtype of the 3D network on CUDA (``None``: float32).

    Returns:
        A :class:`VolumePrediction` (``labels`` is the input, with 3D added).
    """
    dev = _device(device)
    net, config = load_volume_model(volume_model, device=dev)
    vol = volume_head(config)
    part_names = list(vol.part_names)
    center_nodes = list(vol.center_nodes)
    ci_dir = centered_instance_model or vol.centered_instance_model
    if not ci_dir or not os.path.exists(os.path.join(ci_dir, "best.ckpt")):
        raise FileNotFoundError(
            f"The volume's centered-instance model is not at {ci_dir!r}; pass its folder "
            "with -m."
        )
    sha = file_sha256(os.path.join(ci_dir, "best.ckpt"))
    if vol.centered_instance_sha256 and sha != vol.centered_instance_sha256:
        msg = (
            f"{ci_dir} is not the centered-instance model this volume was trained on "
            f"(sha256 {sha[:12]}..., trained on {str(vol.centered_instance_sha256)[:12]}...). "
            "A volume read on another 2D model lost 22-117 % against triangulation in "
            "the study."
        )
        if not allow_other_ci:
            raise ValueError(
                msg + " Pass --allow_other_ci for a retrained 2D model of the same rig."
            )
        logger.warning(msg + " Going on (--allow_other_ci).")
    ci = FrozenCenteredInstance.load(ci_dir, device=dev, sha256=sha)
    builder = VolumeInput(
        ci=ci,
        part_names=part_names,
        n_voxels=int(vol.n_voxels),
        cube_size=float(vol.cube_size),
        image_decode=str(vol.image_decode),
    )
    if passes == "auto":
        n_passes = default_passes(center_nodes, ci.anchor_part)
    else:
        n_passes = int(passes)
        if n_passes not in (1, 2):
            raise ValueError(f"passes must be auto, 1 or 2; got {passes!r}.")
    logger.info(
        f"Volume {volume_model}: {len(part_names)} parts, cube {vol.cube_size} / "
        f"{vol.n_voxels}, centred on {center_nodes}; {n_passes} pass(es); 2D model {ci_dir}"
    )
    if not labels.sessions:
        raise ValueError(
            "centered_volume needs a recording session (calibrated cameras and their "
            "videos) in the input .slp; it has none."
        )
    mv = multiview_config(config)
    out: List[SessionVolume] = []
    net_dtype = net_dtype if dev.type == "cuda" else None
    if dev.type == "cuda":
        net = net.to(memory_format=torch.channels_last_3d)
        torch.backends.cudnn.benchmark = True
    for si, session in enumerate(labels.sessions):
        todo = (
            _frames_without_groups(session, frames)
            if centroid_model is not None
            else []
        )
        if todo:
            _run_2d(
                labels,
                session,
                todo,
                [centroid_model, ci_dir],
                ci.part_names,
                max_animals,
                str(dev),
                batch_size,
            )
        data = MV.extract_session_data(session, part_names, target_source="triangulate")
        if frames is not None and len(data):
            keep = np.isin(data.frame_idx, np.asarray(list(frames)))
            data = _subset(data, keep)
        if not len(data):
            logger.warning(f"Session {si}: no instance groups to run the volume on.")
            continue
        rel = MV.reliable_mask(data.dlt, data.nview, data.err)
        fit = MV.resolve_up(
            session,
            data.cameras,
            data.dlt,
            rel,
            part_names,
            up=list(mv.up) if mv.up is not None else None,
            up_fit_nodes=list(mv.up_fit_nodes) if mv.up_fit_nodes else None,
            band=float(mv.floor_band),
        )
        cen = MV.cube_centers(data.dlt, part_names, center_nodes)
        rows = np.nonzero(np.isfinite(cen).all(-1))[0]
        if len(rows) < len(data):
            logger.info(
                f"Session {si}: {len(data) - len(rows)} of {len(data)} animal-frames have "
                "no cube centre (too few triangulated nodes); skipped."
            )
        S = VolumeSession(
            data=data,
            up=fit.up,
            targets=np.zeros((len(data), len(part_names)), bool),
            reliable=rel,
            centers=cen,
            rows=rows,
        )
        ds = CenteredVolumeDataset(
            [S],
            np.stack([np.zeros(len(rows), np.int64), rows], 1),
            channels=ci.in_channels,
            image_decode=builder.image_decode,
            input_scale=ci.input_scale,
            cache_frames="none",
        )
        res = _run_rows(
            net,
            builder,
            ds,
            part_names,
            center_nodes,
            n_passes,
            batch_size,
            dev,
            net_dtype,
        )
        means = np.full((len(data), len(part_names), 3), np.nan)
        covs = np.full((len(data), len(part_names), 3, 3), np.nan)
        peak = np.full((len(data), len(part_names)), np.nan)
        means[rows], covs[rows], peak[rows] = res
        # cov_scale: measured on this run against the robust DLT, as LUC3D's exporter
        if cov_scale is not None:
            cs, src, npts = float(cov_scale), "given", 0
        else:
            cs, npts = measure_cov_scale(means, covs, data.dlt, data.nview, data.err)
            src = "measured"
            if cs is None:
                cs, src = vol.validation_cov_scale, "validation"
                logger.warning(
                    f"Session {si}: only {npts} points with >= 4 views and < 3 px to "
                    f"measure cov_scale (20 needed); using the model's validation value "
                    f"{cs}."
                )
        logger.info(f"Session {si}: cov_scale {cs} ({src}, {npts} points)")
        # 3D onto the instance groups
        n_set = n_kept = 0
        for r in rows:
            fg = session.frame_groups[int(data.frame_idx[r])]
            ig = fg.instance_groups[int(data.group_index[r])]
            if _set_instance_3d(ig, means[r], peak[r], part_names):
                n_set += 1
            else:
                n_kept += 1
        if n_kept:
            logger.warning(
                f"Session {si}: {n_kept} instance groups keep their user-placed 3D (the "
                "volume's is only in the .vol3d.h5)."
            )
        # frame x animal arrays for the file
        fr = np.unique(data.frame_idx[rows]) if len(rows) else np.zeros(0, np.int64)
        A = int(np.bincount(data.frame_idx[rows]).max()) if len(rows) else 0
        J = len(part_names)
        M = np.full((len(fr), A, J, 3), np.nan)
        C = np.full((len(fr), A, J, 3, 3), np.nan)
        P = np.full((len(fr), A, J), np.nan)
        slot = {}
        for r in rows:
            fi = int(np.searchsorted(fr, data.frame_idx[r]))
            a = slot.get(fi, 0)
            slot[fi] = a + 1
            M[fi, a], C[fi, a], P[fi, a] = means[r], covs[r], peak[r]
        out.append(
            SessionVolume(
                session_index=si,
                node_names=part_names,
                frames=fr,
                means=M,
                covs=C,
                peak=P,
                cov_scale=cs,
                cov_scale_source=src,
                cov_scale_points=npts,
            )
        )
    return VolumePrediction(labels=labels, sessions=out)


def _subset(data: MV.SessionData, keep: np.ndarray) -> MV.SessionData:
    fields = (
        "frame_idx",
        "group_index",
        "pts2d",
        "X",
        "nview",
        "err",
        "given_3d",
        "dlt",
    )
    kw = {f: getattr(data, f)[keep] for f in fields}
    return MV.SessionData(
        session=data.session,
        cameras=data.cameras,
        videos=data.videos,
        node_names=data.node_names,
        **kw,
    )


@torch.no_grad()
def _run_rows(
    net, builder, ds, part_names, center_nodes, n_passes, batch_size, dev, net_dtype
):
    """The volume on every sample of ``ds``, in fixed-size batches."""
    n = len(ds)
    means = np.zeros((n, len(part_names), 3))
    covs = np.zeros((n, len(part_names), 3, 3))
    peak = np.zeros((n, len(part_names)))
    for start in range(0, n, batch_size):
        idx = list(range(start, min(start + batch_size, n)))
        real = len(idx)
        idx += [idx[-1]] * (batch_size - real)  # pad: one batch shape throughout
        batch = collate_volume([ds[i] for i in idx])
        batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
        centers = batch["center"]
        vmask = batch["view_ok"].float()
        res = None
        for _ in range(n_passes):
            res = run_volume(net, builder, batch, centers, vmask, net_dtype)
            centers = centers_from_points(res["means"], part_names, center_nodes)
        means[start : start + real] = res["means"][:real].double().cpu().numpy()
        covs[start : start + real] = res["covs"][:real].cpu().numpy()
        peak[start : start + real] = res["peak"][:real].float().cpu().numpy()
    return means, covs, peak


def write_session_volumes(
    prediction: VolumePrediction,
    path: str,
    source: str,
    chains_path: Optional[str] = None,
) -> List[str]:
    """Write each session's ``.vol3d.h5``.

    One session: ``path``. Several: ``<stem>.<session index>.vol3d.h5`` next to it.

    Returns:
        The files written.
    """
    written = []
    base = str(path)
    stem = (
        base[: -len(".vol3d.h5")]
        if base.endswith(".vol3d.h5")
        else os.path.splitext(base)[0]
    )
    many = len(prediction.sessions) > 1
    for sv in prediction.sessions:
        if sv.cov_scale is None:
            raise ValueError(
                f"Session {sv.session_index}: no cov_scale (too few well-seen points and no "
                "validation value in the model); pass --cov_scale."
            )
        chains = load_chains(chains_path, sv.node_names) if chains_path else ()
        fi, m, c6 = to_rows(sv.frames, sv.means, sv.covs)
        out = f"{stem}.{sv.session_index}.vol3d.h5" if many else f"{stem}.vol3d.h5"
        write_volume3d(out, fi, m, c6, sv.node_names, sv.cov_scale, source, chains)
        logger.info(
            f"Wrote {out}: {len(fi)} animal-frames on {len(np.unique(fi))} frames, "
            f"{len(sv.node_names)} nodes, cov_scale {sv.cov_scale:.4f} ({sv.cov_scale_source})"
        )
        written.append(out)
    return written
