"""Tests for training the volumetric 3D model (`centered_volume`)."""

import json
from pathlib import Path

import numpy as np
import pytest
import sleap_io as sio
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from sleap_nn.config.training_job_config import verify_training_cfg
from sleap_nn.data import multiview as MV
from sleap_nn.data import volume_input as VI
from sleap_nn.data.instance_cropping import make_centered_bboxes
from sleap_nn.inference.ops.crops import crop_bboxes
from sleap_nn.training import volume as TV
from tests.fixtures.multiview import (
    make_session_labels,
    render_png_videos,
    small_cameras,
    two_node_poses,
)

CI_DIR = "tests/assets/model_ckpts/minimal_instance_centered_instance"


def _session_slp(tmp_path, n_frames=8, n_animals=2, seed=0):
    cams = small_cameras(3, seed=seed)
    poses = two_node_poses(n_frames * n_animals, seed=seed).reshape(
        n_frames, n_animals, 2, 3
    )
    vids = render_png_videos(tmp_path / "frames", cams, poses)
    labels, _ = make_session_labels(
        n_frames,
        n_animals,
        cameras=cams,
        videos=vids,
        poses=poses,
        nodes=["A", "B"],
    )
    path = tmp_path / "session.slp"
    labels.save(str(path))
    return path, poses


def _config(tmp_path, train_path, **head):
    vol = {
        "centered_instance_model": CI_DIR,
        "center_nodes": ["A", "B"],
        "n_voxels": 16,
        "ce_sigma_voxels": 1.5,
    }
    vol.update(head)
    return OmegaConf.create(
        {
            "data_config": {
                "train_labels_path": [str(train_path)],
                "validation_fraction": 0.25,
                "multiview": {"min_nodes": 2, "val_samples": 8, "up": [0.0, 0.0, 1.0]},
                "augmentation_config": {"volume": {"view_dropout": 0.3}},
            },
            "model_config": {
                "backbone_config": {"unet3d": {"filters": 4, "max_stride": 4}},
                "head_configs": {"centered_volume": {"volume": vol}},
            },
            "trainer_config": {
                "train_data_loader": {"batch_size": 2, "num_workers": 0},
                "val_data_loader": {"batch_size": 2, "num_workers": 0},
                "max_epochs": 2,
                "train_steps_per_epoch": 2,
                "min_train_steps_per_epoch": 1,
                "save_ckpt": True,
                "ckpt_dir": str(tmp_path / "models"),
                "run_name": "vol",
                "trainer_accelerator": "cpu",
                "enable_progress_bar": False,
                "optimizer_name": "AdamW",
                "optimizer": {"lr": 1e-3},
                "lr_scheduler": {"cosine_annealing_warmup": {"warmup_epochs": 1}},
                "early_stopping": {"stop_training_on_plateau": False},
            },
        }
    )


# ---------------------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------------------


def test_config_validation():
    base = {
        "data_config": {"train_labels_path": ["x.slp"]},
        "model_config": {
            "backbone_config": {"unet3d": {}},
            "head_configs": {
                "centered_volume": {"volume": {"centered_instance_model": "ci"}}
            },
        },
    }
    cfg = verify_training_cfg(OmegaConf.create(base))
    vol = cfg.model_config.head_configs.centered_volume.volume
    assert vol.cube_size == "auto" and vol.n_voxels == 96 and vol.ce_weight == 1.0
    assert cfg.model_config.backbone_config.unet3d.max_stride == 8
    # the backbone and head only go together
    bad = OmegaConf.create(base)
    bad.model_config.backbone_config = {"unet": {}}
    with pytest.raises(ValueError, match="go together"):
        verify_training_cfg(bad)
    bad = OmegaConf.create(base)
    bad.model_config.head_configs = {"centroid": {"confmaps": {}}}
    with pytest.raises(ValueError, match="go together"):
        verify_training_cfg(bad)
    bad = OmegaConf.create(base)
    bad.model_config.head_configs.centered_volume.volume.n_voxels = 20
    with pytest.raises(ValueError, match="divisible"):
        verify_training_cfg(bad)
    bad = OmegaConf.create(base)
    bad.model_config.head_configs.centered_volume.volume.centered_instance_model = None
    with pytest.raises(ValueError, match="centered_instance_model"):
        verify_training_cfg(bad)
    bad = OmegaConf.create(base)
    bad.model_config.head_configs.centered_volume.volume.cube_size = -3
    with pytest.raises(Exception, match="cube_size"):
        verify_training_cfg(bad)
    bad = OmegaConf.create(base)
    bad.model_config.head_configs.centered_volume.volume.unknown_key = 1
    with pytest.raises(Exception):
        verify_training_cfg(bad)


def test_get_train_val_datasets_refuses_volume():
    from sleap_nn.data.custom_datasets import get_train_val_datasets

    cfg = verify_training_cfg(
        OmegaConf.create(
            {
                "data_config": {"train_labels_path": ["x.slp"]},
                "model_config": {
                    "backbone_config": {"unet3d": {}},
                    "head_configs": {
                        "centered_volume": {"volume": {"centered_instance_model": "ci"}}
                    },
                },
            }
        )
    )
    with pytest.raises(ValueError, match="VolumeModelTrainer"):
        get_train_val_datasets(train_labels=[], val_labels=[], config=cfg)


def test_default_passes():
    assert TV.default_passes(["Trunk"], "Trunk") == 1
    assert TV.default_passes(["Neck", "TTI"], "Center") == 2
    assert TV.default_passes(["A", "B"], None) == 2


def test_time_block_split_and_auto_cube_size():
    d = MV.SessionData(
        session=None,
        cameras=None,
        videos=[],
        node_names=["a", "b"],
        frame_idx=np.repeat(np.arange(10), 2),
        group_index=np.tile([0, 1], 10),
        pts2d=np.zeros((20, 1, 2, 2)),
        X=np.zeros((20, 2, 3)),
        nview=np.zeros((20, 2)),
        err=np.zeros((20, 2)),
        given_3d=np.zeros(20, bool),
        dlt=np.zeros((20, 2, 3)),
    )
    rows = np.arange(20)
    tr, va = TV._time_block_split(d, rows, 0.2)
    assert set(d.frame_idx[va]) == {8, 9} and set(d.frame_idx[tr]) == set(range(8))
    tr, va = TV._time_block_split(d, rows, 0.0)
    assert len(va) == 0 and len(tr) == 20
    # SLAP's case: the far node reaches 189 units at the 99th percentile -> 384
    rng = np.random.default_rng(0)
    X = np.zeros((2000, 2, 3))
    X[:, 1, 0] = rng.uniform(0, 189 / 0.99, 2000)
    S = VI.VolumeSession(
        data=MV.SessionData(
            None,
            None,
            [],
            ["a", "b"],
            np.zeros(2000, int),
            np.zeros(2000, int),
            np.zeros((2000, 1, 2, 2)),
            X,
            np.full((2000, 2), 3),
            np.zeros((2000, 2)),
            np.zeros(2000, bool),
            X,
        ),
        up=np.array([0, 0, 1.0]),
        targets=np.ones((2000, 2), bool),
        reliable=np.ones((2000, 2), bool),
        centers=np.zeros((2000, 3)),
        rows=np.arange(2000),
    )
    size = TV._auto_cube_size([S], [np.arange(2000)], 96)
    assert size == pytest.approx(384.0)


# ---------------------------------------------------------------------------------------
# Crops and input
# ---------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ci():
    return VI.FrozenCenteredInstance.load(CI_DIR)


def test_frozen_ci(ci):
    assert ci.part_names == ["A", "B"] and ci.anchor_part is None
    assert ci.crop_size == 96 and ci.input_scale == 1.0 and ci.output_stride == 2
    assert ci.max_stride == 8 and ci.in_channels == 1 and len(ci.sha256) == 64
    assert not any(p.requires_grad for p in ci.model.parameters())
    assert ci.channel_index(["B", "A"]) == [1, 0]
    with pytest.raises(ValueError, match="no maps"):
        ci.channel_index(["C"])


def test_full_crops_match_topdown_inference(ci):
    """The volume's maps are the centered-instance inference layer's for that centroid."""
    from sleap_nn.inference.loaders import _load_lightning_module
    from sleap_nn.inference.topdown import FindInstancePeaks
    from sleap_nn.training.lightning_modules import (
        TopDownCenteredInstanceLightningModule,
    )

    module, _, _ = _load_lightning_module(
        TopDownCenteredInstanceLightningModule,
        CI_DIR,
        model_type="centered_instance",
        device="cpu",
    )
    rng = np.random.default_rng(0)
    img = torch.as_tensor(rng.integers(0, 255, (1, 1, 128, 160)), dtype=torch.uint8)
    for c in ([70.3, 50.8], [5.6, 120.2], [-20.0, 30.0]):
        cen = torch.tensor([c], dtype=torch.float32)
        crops, x0 = VI.crop_views_full(img, cen, ci)
        mine = VI.ci_maps(crops, ci, [0, 1])
        bbox = make_centered_bboxes(cen, 96, 96)
        inst = crop_bboxes(img, bbox, [0])
        layer = FindInstancePeaks(
            torch_model=module,
            output_stride=2,
            return_confmaps=True,
            input_scale=1.0,
            max_stride=8,
        )
        out = layer(
            {
                "instance_image": inst[:, None],
                "eff_scale": torch.ones(1),
                "instance_bbox": bbox[:, None],
            }
        )
        # the same crop as inference, through the same network: exactly its output
        assert torch.equal(VI.crop_views_full(img, cen, ci)[0] * 255, inst.float())
        with torch.no_grad():
            assert torch.equal(mine, module(inst[:, None].clone()))
        # and the inference layer's maps up to CPU convolution noise (it reaches the
        # same module by another memory path: differences of ~1e-6)
        assert torch.allclose(mine, out["pred_confmaps"].float(), atol=1e-5)
        # x0 is crop_bboxes' top-left (its legacy rule truncates toward zero, so an
        # off-image centre at x = -20 starts at -67, not -68)
        assert torch.equal(x0, torch.trunc(cen + 0.5) - 48)


def _study_reduced_crop(x01, uvc, crop):
    """The study's ci_features crop (train_vol.py), for scale 0.5."""
    B, Vn, h, w = x01.shape
    half = crop // 2
    ch = torch.round(uvc / 2)
    ch[..., 0] = ch[..., 0].clamp(-half, w + half)
    ch[..., 1] = ch[..., 1].clamp(-half, h + half)
    ch = torch.nan_to_num(ch, nan=0.0).long()
    x0h = ch - half
    xp = F.pad(x01, (crop, crop, crop, crop))
    crops = torch.stack(
        [
            xp[
                b,
                v,
                int(x0h[b, v, 1]) + crop : int(x0h[b, v, 1]) + 2 * crop,
                int(x0h[b, v, 0]) + crop : int(x0h[b, v, 0]) + 2 * crop,
            ]
            for b in range(B)
            for v in range(Vn)
        ]
    )[:, None]
    return crops, (x0h * 2).float()


def test_reduced_crops_are_the_study_crops():
    ci_half = VI.FrozenCenteredInstance(
        model=None,
        path="",
        sha256="",
        part_names=["A"],
        anchor_part=None,
        crop_size=500,
        input_scale=0.5,
        output_stride=2,
        max_stride=64,
        in_channels=1,
    )
    assert ci_half.reduced_crop == 256  # DoublePoke: 500 -> 250 -> 256
    rng = np.random.default_rng(1)
    img = torch.as_tensor(rng.integers(0, 255, (2, 3, 1, 300, 400)), dtype=torch.uint8)
    uvc = torch.tensor(
        [
            [[10.0, 20.0], [700.3, 590.6], [-900.0, 30.0]],
            [[401.5, 300.5], [float("nan"), 3.0], [799.0, 2000.0]],
        ]
    )
    crops, x0 = VI.crop_views_reduced(
        img.reshape(6, 1, 300, 400), uvc.reshape(6, 2), ci_half
    )
    ref, ref_x0 = _study_reduced_crop(img[:, :, 0].float() / 255.0, uvc, 256)
    assert torch.equal(crops, ref) and torch.equal(x0, ref_x0.reshape(6, 2))


def test_volume_input_chunked_equals_unchunked(ci, monkeypatch):
    cams = MV.CameraArrays.from_cameras(small_cameras(3, seed=2))
    T = {k: v[None].expand(2, *v.shape) for k, v in cams.to_torch().items()}
    rng = np.random.default_rng(3)
    images = torch.as_tensor(
        rng.integers(0, 255, (2, 3, 1, 128, 160)), dtype=torch.uint8
    )
    center = torch.tensor([[0.0, 5.0, 15.0], [10.0, -8.0, 12.0]])
    basis = MV.cube_basis(torch.tensor([[0.0, 0.0, 1.0]] * 2), torch.tensor([0.3, 2.0]))
    vmask = torch.tensor([[1.0, 1.0, 1.0], [1.0, 0.0, 1.0]])
    builder = VI.VolumeInput(ci=ci, part_names=["A", "B"], n_voxels=8, cube_size=80.0)
    a, grid = builder(images, T, center, basis, vmask)
    monkeypatch.setattr(VI, "VIEW_SAMPLE_MAX", 0)
    monkeypatch.setattr(VI, "VIEW_CHUNK_BYTES", 1)
    b, _ = builder(images, T, center, basis, vmask)
    assert a.shape == (2, 2, 8, 8, 8) and grid.shape == (2, 512, 3)
    assert torch.allclose(a, b, atol=1e-6)
    assert a.abs().sum() > 0


def test_frame_encode_decode(tmp_path):
    img = (np.arange(64 * 80) % 251).astype(np.uint8).reshape(64, 80)
    vid = render_png_videos(tmp_path, small_cameras(1), np.zeros((1, 1, 1, 3)))[0]
    buf = VI.encoded_frame(vid, 0)
    full = VI.decode_frame(buf, 1, 1.0)
    assert full.shape == (1, 128, 160)
    half = VI.decode_frame(buf, 1, 0.5)
    assert half.shape == (1, 64, 80)
    rgb = VI.decode_frame(buf, 3, 1.0)
    assert rgb.shape == (3, 128, 160)
    import cv2

    ok, jpg = cv2.imencode(".jpg", img)
    assert VI.decode_frame(jpg.ravel(), 1, 0.5).shape == (1, 32, 40)


# ---------------------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------------------


def test_train_end_to_end(tmp_path):
    from sleap_nn.train import run_training

    path, _ = _session_slp(tmp_path)
    cfg = _config(tmp_path, path)
    trainer = run_training(cfg)
    run = Path(tmp_path) / "models" / "vol"
    assert (run / "best.ckpt").exists() and (run / "training_config.yaml").exists()
    assert (run / "initial_config.yaml").exists() and (run / "metrics.csv").exists()
    saved = OmegaConf.load(run / "training_config.yaml")
    vol = saved.model_config.head_configs.centered_volume.volume
    assert list(vol.part_names) == ["A", "B"] and list(vol.center_nodes) == ["A", "B"]
    assert isinstance(vol.cube_size, float) and vol.cube_size > 0
    assert len(vol.centered_instance_sha256) == 64
    assert saved.model_config.backbone_config.unet3d.in_channels == 2
    split = json.loads((run / "split.json").read_text())
    assert split["train"][0]["frames"] and split["val"][0]["frames"]
    assert max(split["train"][0]["frames"]) < min(split["val"][0]["frames"])
    # the checkpoint holds the 3D network only
    ck = torch.load(run / "best.ckpt", map_location="cpu", weights_only=False)
    assert all(k.startswith("net.") for k in ck["state_dict"])
    assert ck["centered_volume"]["part_names"] == ["A", "B"]
    net, conf = TV.load_volume_model(str(run))
    assert sum(p.numel() for p in net.parameters()) == conf.model_config.total_params
    # the logged validation numbers
    metrics = (run / "metrics.csv").read_text()
    assert "val/median_error" in metrics and "train/loss_step" in metrics


def test_training_refuses_unknown_center(tmp_path):
    path, _ = _session_slp(tmp_path, n_frames=2)
    cfg = _config(tmp_path, path, center_nodes=None)
    with pytest.raises(ValueError, match="center_nodes must be given"):
        TV.train_volume(cfg)
    cfg = _config(tmp_path, path, part_names=["A", "C"])
    with pytest.raises(ValueError, match="no maps"):
        TV.train_volume(cfg)


def test_load_study_checkpoint(tmp_path):
    from tests.fixtures.volume_study import StudyUNet3D

    study = StudyUNet3D(15, 15, filters=8)
    args = dict(filters=8, rate=2.0, down=3, norm="group", n=96, size=384.0, rig="")
    torch.save(
        dict(state=study.state_dict(), args=args, step=8000), tmp_path / "last.pt"
    )
    net, a = TV.load_study_checkpoint(str(tmp_path / "last.pt"))
    assert a["size"] == 384.0
    x = torch.randn(1, 15, 8, 8, 8)
    with torch.no_grad():
        assert torch.equal(net(x), study.eval()(x))
