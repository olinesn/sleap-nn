"""The volumetric model's input: the frozen 2D model's maps, unprojected into a cube.

For each animal (a cube centre, a floor normal and a yaw) and each camera:
    1. the cube centre is projected into the camera and a crop is taken there, as
       sleap-nn's top-down inference does for a centroid at that point;
    2. the frozen centered-instance model gives its confidence maps on the crop;
    3. every voxel is projected into the camera and the maps of ``part_names`` are read
       there (bilinear, zero outside);
then the samples are averaged over the cameras that see each voxel. The result has one
channel per body part, whatever the number of cameras.

Two crop modes (``image_decode`` in the volume head config):
    - ``"full"``: full-resolution frames, cropped with sleap-nn's ``crop_bboxes`` at the
      model's ``crop_size``, then resized by its input scale and padded to its max
      stride: exactly what ``sleap-nn predict`` feeds the model.
    - ``"reduced"``: frames already at the model's input scale (JPEG decoded at reduced
      size), cropped there around the rounded centre: the 2026-11 study's input, kept for
      reproducing its checkpoints bit for bit.

Also here: :class:`FrozenCenteredInstance` (the 2D model), the frame cache, and
:class:`CenteredVolumeDataset` (one sample per instance group per frame group).
"""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import attrs
import cv2
import numpy as np
import sleap_io as sio
import torch
import torch.nn.functional as F
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from sleap_nn.data import multiview as MV
from sleap_nn.data.instance_cropping import make_centered_bboxes
from sleap_nn.data.resizing import apply_pad_to_stride, resize_image
from sleap_nn.inference.ops.crops import crop_bboxes

# Past this many bytes of float32 view samples the unprojection runs a few cameras at a
# time (the study's MultiCam_18 fix: 8 animals x 17 cameras x 34 nodes x 96^3 voxels =
# 15 GiB ran an A100 out of memory). Below it, all cameras at once (the study's path).
VIEW_SAMPLE_MAX = 8 << 30
VIEW_CHUNK_BYTES = 3 << 30


def file_sha256(path: str, chunk: int = 1 << 24) -> str:
    """sha256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------------------------------
# The frozen 2D model
# ---------------------------------------------------------------------------------------


@attrs.define(eq=False)
class FrozenCenteredInstance:
    """A trained centered-instance model, frozen, as the volume's feature extractor.

    Attributes:
        model: The 2D network (eval mode, no gradients): crops to a dict of maps.
        path: Its model folder.
        sha256: sha256 of its ``best.ckpt``.
        part_names: Its body parts, in map order.
        anchor_part: The node its crops are centred on (``None``: the centroid model's
            centre of mass).
        crop_size: Crop side in full-resolution pixels.
        input_scale: Resize factor applied to the crops.
        output_stride: Map stride relative to the resized crop.
        max_stride: The crop is padded to a multiple of this.
        in_channels: Image channels the network takes (1 or 3).
    """

    model: torch.nn.Module
    path: str
    sha256: str
    part_names: List[str]
    anchor_part: Optional[str]
    crop_size: int
    input_scale: float
    output_stride: int
    max_stride: int
    in_channels: int

    @classmethod
    def load(
        cls, model_dir: str, device="cpu", sha256: Optional[str] = None
    ) -> "FrozenCenteredInstance":
        """Load a centered-instance model folder.

        Args:
            model_dir: The folder (``training_config.yaml`` + ``best.ckpt``).
            device: Torch device.
            sha256: The checkpoint's sha256 when already known (skips hashing).

        Returns:
            The frozen model.
        """
        from sleap_nn.config.utils import get_model_type_from_cfg
        from sleap_nn.inference.loaders import _load_lightning_module
        from sleap_nn.training.lightning_modules import (
            TopDownCenteredInstanceLightningModule,
        )

        module, config, _ = _load_lightning_module(
            TopDownCenteredInstanceLightningModule,
            model_dir,
            model_type="centered_instance",
            device=str(device),
        )
        mt = get_model_type_from_cfg(config)
        if mt != "centered_instance":
            raise ValueError(
                f"{model_dir} is a `{mt}` model; the volume reads a `centered_instance` "
                "model's maps."
            )
        model = module.model.eval().to(device)
        for q in model.parameters():
            q.requires_grad_(False)
        cm = config.model_config.head_configs.centered_instance.confmaps
        pre = config.data_config.preprocessing
        bb_type = next(k for k, v in config.model_config.backbone_config.items() if v)
        bb = config.model_config.backbone_config[bb_type]
        ckpt = os.path.join(model_dir, "best.ckpt")
        return cls(
            model=model,
            path=str(Path(model_dir).resolve()),
            sha256=sha256 or (file_sha256(ckpt) if os.path.exists(ckpt) else ""),
            part_names=list(cm.part_names),
            anchor_part=cm.anchor_part,
            crop_size=int(pre.crop_size),
            input_scale=float(pre.scale),
            output_stride=int(cm.output_stride),
            max_stride=int(OmegaConf.select(bb, "max_stride", default=1) or 1),
            in_channels=int(OmegaConf.select(bb, "in_channels", default=1) or 1),
        )

    def channel_index(self, part_names: Sequence[str]) -> List[int]:
        """Map channels of ``part_names``, by name."""
        missing = [n for n in part_names if n not in self.part_names]
        if missing:
            raise ValueError(
                f"The centered-instance model at {self.path} has no maps for {missing} "
                f"(its parts: {self.part_names})."
            )
        return [self.part_names.index(n) for n in part_names]

    @property
    def map_scale(self) -> float:
        """Map pixels per full-resolution pixel."""
        return self.input_scale / self.output_stride

    @property
    def reduced_crop(self) -> int:
        """The crop side at the input scale, padded up to the max stride."""
        c = int(self.crop_size * self.input_scale)
        return -(-c // self.max_stride) * self.max_stride

    def to(self, device) -> "FrozenCenteredInstance":
        """Move the network."""
        self.model.to(device)
        return self


# ---------------------------------------------------------------------------------------
# Crops and maps
# ---------------------------------------------------------------------------------------


def _photometric(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (x * a + b).clamp_(0, 1)


def crop_views_full(
    images: torch.Tensor,
    centers_px: torch.Tensor,
    ci: FrozenCenteredInstance,
    image_hw: Optional[torch.Tensor] = None,
    photo: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Crop full-resolution frames as sleap-nn's top-down inference does.

    Args:
        images: ``(N, C, H, W)`` uint8 or [0, 1] float frames.
        centers_px: ``(N, 2)`` crop centres in pixels.
        ci: The 2D model (crop size, scale, max stride).
        image_hw: ``(N, 2)`` true image sizes (frames padded to a common size); the
            photometric change applies inside them only.
        photo: Optional per-image ``(a (N,), b (N,))`` brightness/contrast change.

    Returns:
        A tuple ``(crops (N, C, h, w) float in [0, 1], x0 (N, 2) full-res top-left)``.
    """
    N = images.shape[0]
    c = torch.nan_to_num(centers_px.float(), nan=0.0)
    bboxes = make_centered_bboxes(c, ci.crop_size, ci.crop_size)
    crops = crop_bboxes(images, bboxes, torch.arange(N, device=images.device))
    half = torch.tensor(
        [ci.crop_size // 2, ci.crop_size // 2], device=bboxes.device, dtype=bboxes.dtype
    )
    # crop_bboxes' own top-left rule (legacy parity), so the maps line up with it
    x0 = (bboxes[:, 0, :] + half).to(torch.long) - half.long()
    if photo is None:
        # sleap-nn's inference order: resize and pad the crop in its own dtype, then
        # normalize (FindInstancePeaks, then the module's normalize_on_gpu)
        if ci.input_scale != 1.0:
            crops = resize_image(crops, ci.input_scale)
        crops = apply_pad_to_stride(crops, ci.max_stride)
        crops = crops.float()
        if not torch.is_floating_point(images):
            crops = crops / 255.0
        return crops, x0.float()
    crops = crops.float()
    if not torch.is_floating_point(images):
        crops = crops / 255.0
    if photo is not None:
        a, b = photo
        ys = x0[:, 1, None] + torch.arange(ci.crop_size, device=crops.device)[None]
        xs = x0[:, 0, None] + torch.arange(ci.crop_size, device=crops.device)[None]
        H, W = images.shape[-2:]
        hh = image_hw[:, 0, None] if image_hw is not None else torch.full_like(ys, H)
        ww = image_hw[:, 1, None] if image_hw is not None else torch.full_like(xs, W)
        inside = ((ys >= 0) & (ys < hh))[:, :, None] & ((xs >= 0) & (xs < ww))[:, None]
        adj = _photometric(crops, a[:, None, None, None], b[:, None, None, None])
        crops = torch.where(inside[:, None], adj, crops)
    if ci.input_scale != 1.0:
        crops = resize_image(crops, ci.input_scale)
    crops = apply_pad_to_stride(crops, ci.max_stride)
    return crops, x0.float()


def crop_views_reduced(
    images: torch.Tensor,
    centers_px: torch.Tensor,
    ci: FrozenCenteredInstance,
    photo: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Crop frames already at the model's input scale (the study's ``ci_features``).

    Args:
        images: ``(N, C, h, w)`` uint8 or [0, 1] float frames at the input scale.
        centers_px: ``(N, 2)`` crop centres in full-resolution pixels.
        ci: The 2D model.
        photo: Optional per-image ``(a (N,), b (N,))`` brightness/contrast change,
            applied to the whole frame before cropping.

    Returns:
        A tuple ``(crops (N, C, crop, crop) float, x0 (N, 2) full-res top-left)``.
    """
    x = images.float()
    if not torch.is_floating_point(images):
        x = x / 255.0
    if photo is not None:
        a, b = photo
        x = _photometric(x, a[:, None, None, None], b[:, None, None, None])
    N, C, h, w = x.shape
    crop = ci.reduced_crop
    half = crop // 2
    ch = torch.round(centers_px * ci.input_scale)
    ch[..., 0] = ch[..., 0].clamp(-half, w + half)
    ch[..., 1] = ch[..., 1].clamp(-half, h + half)
    ch = torch.nan_to_num(ch, nan=0.0).long()
    x0h = ch - half
    xp = F.pad(x, (crop, crop, crop, crop))
    crops = torch.stack(
        [
            xp[
                i,
                :,
                int(x0h[i, 1]) + crop : int(x0h[i, 1]) + 2 * crop,
                int(x0h[i, 0]) + crop : int(x0h[i, 0]) + 2 * crop,
            ]
            for i in range(N)
        ]
    )
    return crops, x0h.float() / ci.input_scale


def ci_maps(
    crops: torch.Tensor, ci: FrozenCenteredInstance, channels: Sequence[int]
) -> torch.Tensor:
    """The 2D model's maps of the chosen channels, ``(N, K, hm, wm)`` float32."""
    use_amp = crops.is_cuda
    with (
        torch.no_grad(),
        torch.autocast(
            device_type=crops.device.type, dtype=torch.bfloat16, enabled=use_amp
        ),
    ):
        out = ci.model(crops)
    cm = next(iter(out.values())).float()
    return cm[:, list(channels)]


@attrs.define(eq=False)
class VolumeInput:
    """Builds the 3D network's input from camera frames (see the module docstring).

    Attributes:
        ci: The frozen 2D model.
        part_names: The volume's channels, read from the 2D maps by name.
        n_voxels: Voxels per side.
        cube_size: Cube side (world units).
        image_decode: ``"full"`` or ``"reduced"``.
    """

    ci: FrozenCenteredInstance
    part_names: List[str]
    n_voxels: int
    cube_size: float
    image_decode: str = "full"
    channels: List[int] = attrs.field(init=False)

    def __attrs_post_init__(self):
        """Resolve the channels."""
        self.channels = self.ci.channel_index(self.part_names)

    def grid(self, center: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
        """``(B, M, 3)`` voxel centres."""
        offs = MV.cube_offsets(self.n_voxels, self.cube_size, center.device)
        return MV.cube_world(center, basis, offs)

    def _maps(self, images, uvc, vmask_any, image_hw, photo):
        """Crops and maps for the (b, v) pairs of a view chunk -> maps, x0."""
        B, V = images.shape[:2]
        flat = images.reshape(B * V, *images.shape[2:])
        c = uvc.reshape(B * V, 2)
        ph = None
        if photo is not None:
            ph = (photo[0].reshape(-1), photo[1].reshape(-1))
        if self.image_decode == "reduced":
            crops, x0 = crop_views_reduced(flat, c, self.ci, ph)
        else:
            hw = image_hw.reshape(B * V, 2) if image_hw is not None else None
            crops, x0 = crop_views_full(flat, c, self.ci, hw, ph)
        cm = ci_maps(crops, self.ci, self.channels)
        return cm.reshape(B, V, *cm.shape[1:]), x0.reshape(B, V, 2)

    def __call__(
        self,
        images: torch.Tensor,
        cams: Dict[str, torch.Tensor],
        center: torch.Tensor,
        basis: torch.Tensor,
        view_mask: torch.Tensor,
        image_hw: Optional[torch.Tensor] = None,
        photo: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build the input.

        Args:
            images: ``(B, V, C, H, W)`` frames (uint8, or [0, 1] float), at full
                resolution or at the input scale (``image_decode``).
            cams: ``K, dist, R, t`` tensors with a ``(B, V)`` prefix.
            center: ``(B, 3)`` cube centres.
            basis: ``(B, 3, 3)`` cube bases.
            view_mask: ``(B, V)`` 1 for cameras in use.
            image_hw: ``(B, V, 2)`` true frame sizes (frames padded to a common size).
            photo: Optional ``(a (B, V), b (B, V))`` brightness/contrast change.

        Returns:
            A tuple ``(input (B, K, n, n, n), grid (B, M, 3))``.
        """
        B, V = images.shape[:2]
        n = self.n_voxels
        grid = self.grid(center, basis)
        uv, z = MV.project_points(grid, cams["K"], cams["dist"], cams["R"], cams["t"])
        uvc, _ = MV.project_points(
            center[:, None, :], cams["K"], cams["dist"], cams["R"], cams["t"]
        )
        uvc = uvc[:, :, 0]
        K = len(self.channels)
        per_view = B * K * grid.shape[1] * 4
        sc = self.ci.map_scale
        if per_view * V <= VIEW_SAMPLE_MAX:
            cm, x0 = self._maps(images, uvc, None, image_hw, photo)
            origin = MV.map_origin(x0, sc)
            vals, valid = MV.sample_views(
                cm,
                uv,
                z,
                origin,
                torch.full((B, V), sc, device=cm.device),
                cm.shape[-2:],
            )
            agg = MV.average_views(vals, valid, view_mask)
            return agg.reshape(B, K, n, n, n), grid
        # a few cameras at a time, only those some animal uses
        keep = torch.nonzero(view_mask.sum(0) > 0).flatten().tolist() or list(range(V))
        vc = max(1, VIEW_CHUNK_BYTES // per_view)
        num = den = None
        for i in range(0, len(keep), vc):
            vs = keep[i : i + vc]
            ph = None if photo is None else (photo[0][:, vs], photo[1][:, vs])
            hw = None if image_hw is None else image_hw[:, vs]
            cm, x0 = self._maps(images[:, vs], uvc[:, vs], None, hw, ph)
            origin = MV.map_origin(x0, sc)
            vals, valid = MV.sample_views(
                cm,
                uv[:, vs],
                z[:, vs],
                origin,
                torch.full((B, len(vs)), sc, device=cm.device),
                cm.shape[-2:],
            )
            w = valid.float() * view_mask[:, vs, None]
            s_, d_ = (vals * w[:, :, None]).sum(1), w.sum(1)
            num = s_ if num is None else num + s_
            den = d_ if den is None else den + d_
            del cm, vals, valid, w, s_
        agg = num / den.clamp(min=1)[:, None]
        return agg.reshape(B, K, n, n, n), grid


# ---------------------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------------------


def encoded_frame(video: sio.Video, frame_idx: int, quality: int = 95) -> np.ndarray:
    """A frame as encoded bytes (``uint8`` array).

    Frames a video already stores as JPEG or PNG (sleap-io's embedded HDF5 frames, the
    study's frame files) keep their bytes; others are read and encoded as JPEG.
    """
    backend = getattr(video, "backend", None)
    getter = getattr(backend, "get_frame_raw_bytes", None)
    if getter is not None:
        try:
            raw = getter(frame_idx)
        except Exception:  # noqa: BLE001 - fall back to decoding
            raw = None
        if raw is not None:
            return np.frombuffer(np.asarray(raw).tobytes(), np.uint8)
    img = np.asarray(video[frame_idx])
    if img.ndim == 3 and img.shape[-1] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif img.ndim == 3:
        img = img[..., 0]
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError(f"Could not encode frame {frame_idx} of {video.filename}.")
    return buf.ravel()


class FrameStore:
    """Encoded frames of one camera of one session, in one buffer.

    One ``uint8`` buffer and an offset table rather than a numpy array per frame:
    dataloader workers fork with the cache, and a buffer per frame would put object
    headers on the same pages as frame bytes, so reference counting in each worker
    would copy the whole cache page by page (the study kept one buffer per camera
    for this reason).
    """

    def __init__(self, frames: Sequence[int], bufs: Sequence[np.ndarray]):
        """Concatenate ``bufs`` (the encoded ``frames``)."""
        lens = np.array([len(b) for b in bufs], np.int64)
        self.offsets = np.concatenate([[0], np.cumsum(lens)])
        self.data = (
            np.concatenate([np.asarray(b, np.uint8).ravel() for b in bufs])
            if len(bufs)
            else np.zeros(0, np.uint8)
        )
        self.index = {int(f): i for i, f in enumerate(frames)}

    def get(self, frame_idx: int) -> Optional[np.ndarray]:
        """The encoded bytes of a frame (``None`` if not stored)."""
        i = self.index.get(int(frame_idx))
        if i is None:
            return None
        return self.data[self.offsets[i] : self.offsets[i + 1]]


def read_encoded_frames(
    video: sio.Video, frames: Sequence[int], quality: int = 95
) -> List[np.ndarray]:
    """Encoded bytes of many frames of a video.

    Frames embedded as JPEG/PNG in an HDF5 video (sleap-io's embedded layout, the
    study's frame files) are read in one call per video and keep their bytes; other
    videos are read frame by frame and encoded (:func:`encoded_frame`).
    """
    b = getattr(video, "backend", None)
    if type(b).__name__ == "HDF5Video" and getattr(b, "has_embedded_images", False):
        try:
            import h5py

            fmap = b.frame_map or {}
            idx = np.array(
                [fmap.get(int(f), int(f)) if fmap else int(f) for f in frames]
            )
            order = np.argsort(idx, kind="stable")
            with h5py.File(b.filename, "r") as h:
                ds = h[b.dataset]
                uniq, inv = np.unique(idx[order], return_inverse=True)
                data = ds[uniq]
            out: List[Optional[np.ndarray]] = [None] * len(frames)
            for k, o in enumerate(order):
                out[o] = np.asarray(data[inv[k]]).view(np.uint8).ravel()
            return out  # type: ignore[return-value]
        except Exception as e:  # noqa: BLE001 - fall back to frame by frame
            logger.warning(
                f"Batch read of {video.filename} failed ({e}); frame by frame."
            )
    return [encoded_frame(video, int(f), quality) for f in frames]


_REDUCED_FLAGS = {
    (1, 0.5): cv2.IMREAD_REDUCED_GRAYSCALE_2,
    (1, 0.25): cv2.IMREAD_REDUCED_GRAYSCALE_4,
    (1, 0.125): cv2.IMREAD_REDUCED_GRAYSCALE_8,
    (3, 0.5): cv2.IMREAD_REDUCED_COLOR_2,
    (3, 0.25): cv2.IMREAD_REDUCED_COLOR_4,
    (3, 0.125): cv2.IMREAD_REDUCED_COLOR_8,
}


def decode_frame(buf: np.ndarray, channels: int = 1, scale: float = 1.0) -> np.ndarray:
    """Decode bytes to ``(C, h, w)`` uint8, optionally at a reduced scale.

    At 1/2, 1/4 and 1/8 a JPEG decodes at that size directly (DCT scaling, the study's
    input); other formats and scales decode at full size and are resized (area).
    """
    flag = _REDUCED_FLAGS.get((channels, float(scale)))
    is_jpeg = len(buf) > 2 and buf[0] == 0xFF and buf[1] == 0xD8
    if scale != 1.0 and flag is not None and is_jpeg:
        img = cv2.imdecode(buf, flag)
    else:
        img = cv2.imdecode(
            buf, cv2.IMREAD_GRAYSCALE if channels == 1 else cv2.IMREAD_COLOR
        )
        if scale != 1.0:
            h, w = img.shape[:2]
            img = cv2.resize(
                img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
            )
    if img.ndim == 2:
        return img[None]
    return np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1))


def read_frame(video: sio.Video, frame_idx: int, channels: int = 1) -> np.ndarray:
    """Read a frame from a video as ``(C, H, W)`` uint8 in ``channels`` channels."""
    img = np.asarray(video[frame_idx])
    if img.ndim == 2:
        img = img[..., None]
    if channels == 1 and img.shape[-1] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)[..., None]
    elif channels == 3 and img.shape[-1] == 1:
        img = np.repeat(img, 3, -1)
    return np.ascontiguousarray(img.transpose(2, 0, 1))


# ---------------------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------------------


@attrs.define(eq=False)
class VolumeSession:
    """One session's training view: its data, floor normal and usable rows."""

    data: MV.SessionData
    up: np.ndarray
    targets: np.ndarray  # (N, J) train mask
    reliable: np.ndarray  # (N, J)
    centers: np.ndarray  # (N, 3) cube centres from the targets
    rows: np.ndarray  # rows with enough targets and a centre


def target_centers(
    X: np.ndarray, mask: np.ndarray, node_names: Sequence[str], center_nodes
) -> np.ndarray:
    """Cube centres from targets: the mean of ``center_nodes``, NaN unless all count.

    The study's training rule (no fallback to other nodes).
    """
    idx = [list(node_names).index(n) for n in center_nodes]
    Xt = np.where(mask[..., None], X, np.nan)
    return Xt[:, idx].mean(1)


class CenteredVolumeDataset(torch.utils.data.Dataset):
    """One sample per instance group on a frame group, with every camera's frame.

    Each item: ``images`` (V, C, h, w) uint8, ``image_hw`` (V, 2), ``view_ok`` (V,),
    the cameras (``K``, ``dist``, ``R``, ``t``), ``up`` (3,), ``center`` (3,), ``X``
    (J, 3) targets (NaN where a node does not train), ``reliable`` (J,), and its
    ``session``, ``frame_idx`` and ``group_index``.

    Args:
        sessions: The sessions (:class:`VolumeSession`).
        samples: ``(N, 2)`` (session, row) pairs.
        channels: Image channels of the 2D model.
        image_decode: ``"full"`` or ``"reduced"``.
        input_scale: The 2D model's input scale (``"reduced"`` decodes at it).
        cache_frames: ``"memory"`` or ``"none"``.
        jpeg_quality: Quality of re-encoded cached frames.
    """

    def __init__(
        self,
        sessions: List[VolumeSession],
        samples: np.ndarray,
        channels: int = 1,
        image_decode: str = "full",
        input_scale: float = 1.0,
        cache_frames: str = "memory",
        jpeg_quality: int = 95,
    ):
        """Index the samples and, with ``cache_frames="memory"``, read their frames."""
        self.sessions = sessions
        self.samples = np.asarray(samples, np.int64).reshape(-1, 2)
        self.channels = channels
        self.image_decode = image_decode
        self.input_scale = input_scale
        self.cache_frames = cache_frames
        self.cache: Dict[Tuple[int, int], FrameStore] = {}
        # without a cache, the frames of the last frame index read (several animals
        # share a frame: each camera frame is decoded once, not once per animal)
        self._last: Dict[Tuple[int, int], List[Optional[np.ndarray]]] = {}
        if cache_frames == "memory":
            self._fill_cache(jpeg_quality)

    def _fill_cache(self, quality: int):
        need = {}
        for si, r in self.samples:
            f = int(self.sessions[si].data.frame_idx[r])
            need.setdefault(int(si), set()).add(f)
        n, nbytes = 0, 0
        for si, frames in need.items():
            for v, video in enumerate(self.sessions[si].data.videos):
                if video is None:
                    continue
                fr = sorted(frames)
                store = FrameStore(fr, read_encoded_frames(video, fr, quality))
                self.cache[(si, v)] = store
                n += len(fr)
                nbytes += store.data.nbytes
        logger.info(f"Cached {n} camera frames in memory ({nbytes / 2**30:.2f} GiB).")

    def __len__(self) -> int:
        """Number of samples."""
        return len(self.samples)

    def _frame(self, si: int, v: int, f: int) -> Optional[np.ndarray]:
        scale = self.input_scale if self.image_decode == "reduced" else 1.0
        if self.cache_frames == "memory":
            store = self.cache.get((si, v))
            buf = store.get(f) if store is not None else None
            if buf is None:
                return None
            return decode_frame(buf, self.channels, scale)
        video = self.sessions[si].data.videos[v]
        if video is None:
            return None
        img = read_frame(video, f, self.channels)
        if scale != 1.0:
            hwc = img.transpose(1, 2, 0)
            h, w = hwc.shape[:2]
            hwc = cv2.resize(
                hwc, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
            )
            img = hwc[None] if hwc.ndim == 2 else hwc.transpose(2, 0, 1)
        return img

    def __getitem__(self, i: int) -> Dict:
        """One sample (see the class docstring)."""
        si, r = (int(x) for x in self.samples[i])
        S = self.sessions[si]
        d = S.data
        f = int(d.frame_idx[r])
        V = d.cameras.n_views
        if self.cache_frames == "memory":
            imgs = [self._frame(si, v, f) for v in range(V)]
        else:
            imgs = self._last.get((si, f))
            if imgs is None:
                imgs = [self._frame(si, v, f) for v in range(V)]
                self._last = {(si, f): imgs}
        ok = np.array([im is not None for im in imgs])
        shapes = [im.shape[1:] for im in imgs if im is not None]
        H = max((s[0] for s in shapes), default=1)
        W = max((s[1] for s in shapes), default=1)
        out = np.zeros((V, self.channels, H, W), np.uint8)
        hw = np.zeros((V, 2), np.int64)
        for v, im in enumerate(imgs):
            if im is not None:
                out[v, :, : im.shape[1], : im.shape[2]] = im
                hw[v] = im.shape[1:]
        X = np.where(S.targets[r][:, None], d.X[r], np.nan).astype(np.float32)
        return dict(
            images=torch.from_numpy(out),
            image_hw=torch.from_numpy(hw),
            view_ok=torch.from_numpy(ok),
            K=torch.from_numpy(d.cameras.K.astype(np.float32)),
            dist=torch.from_numpy(d.cameras.dist.astype(np.float32)),
            R=torch.from_numpy(d.cameras.R.astype(np.float32)),
            t=torch.from_numpy(d.cameras.t.astype(np.float32)),
            up=torch.from_numpy(S.up.astype(np.float32)),
            center=torch.from_numpy(S.centers[r].astype(np.float32)),
            X=torch.from_numpy(X),
            reliable=torch.from_numpy(S.reliable[r]),
            session=si,
            frame_idx=f,
            group_index=int(d.group_index[r]),
        )


def collate_volume(items: List[Dict]) -> Dict:
    """Stack samples, padding cameras (``view_ok`` False) and frame sizes (zeros)."""
    V = max(it["images"].shape[0] for it in items)
    C = items[0]["images"].shape[1]
    H = max(it["images"].shape[2] for it in items)
    W = max(it["images"].shape[3] for it in items)
    B = len(items)
    out = {
        "images": torch.zeros((B, V, C, H, W), dtype=torch.uint8),
        "image_hw": torch.zeros((B, V, 2), dtype=torch.long),
        "view_ok": torch.zeros((B, V), dtype=torch.bool),
    }
    eye = torch.eye(3)
    cams = {
        "K": eye.repeat(B, V, 1, 1),
        "dist": torch.zeros(B, V, 5),
        "R": eye.repeat(B, V, 1, 1),
        "t": torch.zeros(B, V, 3),
    }
    cams["t"][..., 2] = 1.0  # padded cameras: harmless, and masked out
    for b, it in enumerate(items):
        v, _, h, w = it["images"].shape
        out["images"][b, :v, :, :h, :w] = it["images"]
        out["image_hw"][b, :v] = it["image_hw"]
        out["view_ok"][b, :v] = it["view_ok"]
        for k in cams:
            cams[k][b, :v] = it[k]
    out.update(cams)
    for k in ("up", "center", "X", "reliable"):
        out[k] = torch.stack([it[k] for it in items])
    for k in ("session", "frame_idx", "group_index"):
        out[k] = torch.tensor([it[k] for it in items], dtype=torch.long)
    return out
