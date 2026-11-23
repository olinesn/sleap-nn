"""End-to-end tests of the volumetric 3D model: train, predict, write, read back.

A 4-camera synthetic session (frames with a bright spot per keypoint), the repo's tiny
centered-instance model as the frozen 2D model, a few training steps, then the Python
API and ``sleap-nn predict``: the ``.slp`` gets 3D on its instance groups and LUC3D's
``.vol3d.h5`` is written; both are read back.
"""

import json

import attrs
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest
import sleap_io as sio
import torch
from click.testing import CliRunner

from sleap_nn.cli import cli
from sleap_nn.inference.volume import predict_volume3d, write_session_volumes
from sleap_nn.io.volume3d import FORMAT, read_volume3d, unpack_covariances
from sleap_nn.training.volume import train_volume
from tests.fixtures.multiview import (
    make_session_labels,
    render_png_videos,
    small_cameras,
    two_node_poses,
)
from tests.training.test_volume_training import CI_DIR, _config


def _session(tmp_path, n_frames=6, n_animals=2, seed=0, name="session.slp"):
    cams = small_cameras(4, seed=seed)
    poses = two_node_poses(n_frames * n_animals, seed=seed).reshape(
        n_frames, n_animals, 2, 3
    )
    vids = render_png_videos(tmp_path / f"frames_{seed}", cams, poses)
    labels, _ = make_session_labels(
        n_frames, n_animals, cameras=cams, videos=vids, poses=poses, nodes=["A", "B"]
    )
    path = tmp_path / name
    labels.save(str(path))
    return path, poses


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """A tiny trained volume and the session it was trained on."""
    tmp = tmp_path_factory.mktemp("volume")
    path, poses = _session(tmp)
    train_volume(_config(tmp, path))
    return tmp / "models" / "vol", path, poses


def test_predict_volume3d_python_api(trained):
    model, path, poses = trained
    labels = sio.load_slp(str(path))
    pred = predict_volume3d(labels, str(model), device="cpu", batch_size=4)
    assert len(pred.sessions) == 1
    sv = pred.sessions[0]
    assert sv.node_names == ["A", "B"]
    assert sv.frames.tolist() == list(range(6))
    assert sv.means.shape == (6, 2, 2, 3) and sv.covs.shape == (6, 2, 2, 3, 3)
    assert np.isfinite(sv.means).all() and np.isfinite(sv.covs).all()
    # covariances are symmetric and positive definite
    assert np.allclose(sv.covs, np.swapaxes(sv.covs, -1, -2))
    assert (np.linalg.eigvalsh(sv.covs) > 0).all()
    # measured on this run: 4 cameras give well-seen points
    assert sv.cov_scale_source == "measured" and sv.cov_scale > 0
    assert sv.cov_scale_points == 6 * 2 * 2
    # every point is inside its cube (half a side from the DLT centre)
    cfg_size = 0.5 * float(
        __import__("omegaconf")
        .OmegaConf.load(model / "training_config.yaml")
        .model_config.head_configs.centered_volume.volume.cube_size
    )
    centre = poses.mean(2)  # the cube centre: midpoint of A and B
    assert (np.abs(sv.means - centre[:, :, None]) <= cfg_size * 1.0001).all()
    # the .slp side: a PredictedInstance3D on every group
    session = pred.labels.sessions[0]
    for f, fg in session.frame_groups.items():
        for ig in fg.instance_groups:
            assert isinstance(ig.instance_3d, sio.PredictedInstance3D)
            assert ig.instance_3d.points.shape == (2, 3)
            assert np.isfinite(ig.instance_3d.point_scores).all()


def test_passes_and_frames(trained):
    model, path, _ = trained
    one = predict_volume3d(sio.load_slp(str(path)), str(model), passes=1, device="cpu")
    two = predict_volume3d(sio.load_slp(str(path)), str(model), passes=2, device="cpu")
    assert not np.allclose(one.sessions[0].means, two.sessions[0].means)
    sub = predict_volume3d(
        sio.load_slp(str(path)), str(model), frames=[1, 3], device="cpu"
    )
    assert sub.sessions[0].frames.tolist() == [1, 3]
    with pytest.raises(ValueError, match="passes"):
        predict_volume3d(sio.load_slp(str(path)), str(model), passes=3, device="cpu")


def test_refuses_another_2d_model(trained, tmp_path):
    model, path, _ = trained
    other = tmp_path / "ci_copy"
    shutil.copytree(CI_DIR, other)
    ck = torch.load(other / "best.ckpt", map_location="cpu", weights_only=False)
    ck["note"] = "re-saved: same weights, different bytes"
    torch.save(ck, other / "best.ckpt")
    with pytest.raises(ValueError, match="allow_other_ci"):
        predict_volume3d(
            sio.load_slp(str(path)),
            str(model),
            centered_instance_model=str(other),
            device="cpu",
        )
    pred = predict_volume3d(
        sio.load_slp(str(path)),
        str(model),
        centered_instance_model=str(other),
        allow_other_ci=True,
        device="cpu",
    )
    assert np.isfinite(pred.sessions[0].means).all()


def test_refuses_labels_without_session(trained):
    model, _, _ = trained
    with pytest.raises(ValueError, match="recording session"):
        predict_volume3d(sio.Labels(), str(model), device="cpu")


def test_keeps_user_placed_3d(trained, tmp_path):
    model, path, poses = trained
    labels = sio.load_slp(str(path))
    ig = labels.sessions[0].frame_groups[0].instance_groups[0]
    skel = ig.instances[0].skeleton
    ig._instance_3d = sio.Instance3D(points=poses[0, 0], skeleton=skel)
    pred = predict_volume3d(labels, str(model), device="cpu")
    kept = pred.labels.sessions[0].frame_groups[0].instance_groups[0].instance_3d
    assert type(kept) is sio.Instance3D and np.allclose(kept.points, poses[0, 0])
    # its volume result is still in the file arrays
    assert np.isfinite(pred.sessions[0].means[0, 0]).all()


def test_write_session_volumes_one_file_per_session(trained, tmp_path):
    model, path, _ = trained
    pred = predict_volume3d(sio.load_slp(str(path)), str(model), device="cpu")
    pred.sessions.append(attrs.evolve(pred.sessions[0], session_index=1))
    files = write_session_volumes(pred, str(tmp_path / "out.vol3d.h5"), "test")
    assert [Path(f).name for f in files] == ["out.0.vol3d.h5", "out.1.vol3d.h5"]
    pred.sessions[0].cov_scale = None
    with pytest.raises(ValueError, match="cov_scale"):
        write_session_volumes(pred, str(tmp_path / "x.vol3d.h5"), "test")


def test_cli_predict_writes_slp_and_vol3d(trained, tmp_path):
    model, path, _ = trained
    chains = tmp_path / "chains.json"
    chain = [{"name": "body", "bones": [["A", "B"]], "nodes": ["B"]}]
    chains.write_text(json.dumps(chain))
    out = tmp_path / "out.slp"
    res = CliRunner().invoke(
        cli,
        [
            "predict",
            "-i",
            str(path),
            "-m",
            str(model),
            "-o",
            str(out),
            "--device",
            "cpu",
            "--volume_passes",
            "2",
            "--chains",
            str(chains),
        ],
        catch_exceptions=False,
    )
    assert res.exit_code == 0, res.output
    # the .slp, read back with sleap-io
    loaded = sio.load_slp(str(out), open_videos=False)
    groups = [
        ig
        for fg in loaded.sessions[0].frame_groups.values()
        for ig in fg.instance_groups
    ]
    assert len(groups) == 12
    assert all(isinstance(ig.instance_3d, sio.PredictedInstance3D) for ig in groups)
    # LUC3D's file, read back with h5py and with the reader
    vol = tmp_path / "out.vol3d.h5"
    with h5py.File(vol, "r") as h:
        assert h.attrs["format"] == FORMAT
        assert h["frame_indices"].dtype == np.int64 and h["means"].dtype == np.float32
        assert h["means"].shape == (12, 2, 3) and h["covs"].shape == (12, 2, 6)
    r = read_volume3d(str(vol))
    assert r["node_names"] == ["A", "B"]
    assert sorted(r["frame_indices"].tolist()) == sorted(list(range(6)) * 2)
    assert np.isfinite(r["cov_scale"]) and r["cov_scale"] > 0
    assert "centered_volume" in r["source"] and r["skeleton_chains"] == chain
    C = unpack_covariances(r["covs"])
    assert (np.linalg.eigvalsh(C.astype(np.float64)) > 0).all()
    # the 3D in the .slp and in the file agree
    pts = np.stack([ig.instance_3d.points for ig in groups])
    assert np.allclose(
        np.sort(pts.ravel()), np.sort(r["means"].astype(np.float64).ravel()), atol=1e-3
    )


def test_cli_refuses_overwriting_input_and_video_input(trained):
    model, path, _ = trained
    runner = CliRunner()
    res = runner.invoke(
        cli, ["predict", "-i", str(path), "-m", str(model), "-o", str(path)]
    )
    assert res.exit_code != 0 and "overwrite" in res.output
    res = runner.invoke(cli, ["predict", "-i", "video.mp4", "-m", str(model)])
    assert res.exit_code != 0 and "recording session" in res.output


def test_2d_path_runs_on_a_session_without_groups(trained, tmp_path):
    """Without instance groups, the 2D models run on every camera and are matched."""
    model, path, _ = trained
    labels = sio.load_slp(str(path))
    session = labels.sessions[0]
    session.frame_groups.clear()
    labels.labeled_frames.clear()
    pred = predict_volume3d(
        labels,
        str(model),
        centroid_model="tests/assets/model_ckpts/minimal_instance_centroid",
        frames=[0, 1],
        max_animals=2,
        device="cpu",
    )
    # the tiny fixture models are not trained on these frames, so how many animals
    # match is not the point: the 2D ran on each camera and nothing broke
    assert len(labels.labeled_frames) > 0
    for sv in pred.sessions:
        assert set(sv.frames.tolist()) <= {0, 1}


def test_sample_config_is_valid():
    from omegaconf import OmegaConf

    from sleap_nn.config.training_job_config import verify_training_cfg

    cfg = verify_training_cfg(
        OmegaConf.load("docs/sample_configs/config_centered_volume_unet3d.yaml")
    )
    vol = cfg.model_config.head_configs.centered_volume.volume
    assert vol.n_voxels == 96 and vol.cube_size == "auto" and vol.ce_weight == 1.0
    assert (
        cfg.trainer_config.max_epochs * cfg.trainer_config.train_steps_per_epoch == 8000
    )
