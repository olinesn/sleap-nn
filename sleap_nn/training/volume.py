"""Training the volumetric 3D model (model type ``centered_volume``).

``sleap-nn train`` hands a ``centered_volume`` config to :class:`VolumeModelTrainer`, a
:class:`~sleap_nn.training.model_trainer.ModelTrainer` whose data, model and evaluation
steps are multi-camera: samples are instance groups on frame groups of recording
sessions, not labeled frames, and the 2D post-training evaluation does not apply. The
2D trainer's code paths are untouched.

The recipe is the 2026-11-11 volumetric study's (``train_vol.py``): AdamW, a per-step
linear warmup and cosine decay, gradient clipping at 1.0, bf16 autocast on CUDA, L1 on
the soft-argmax point plus a volumetric cross-entropy, cube-centre jitter, random yaw,
camera dropout and a per-camera brightness/contrast change.
"""

from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import attrs
import lightning as L
import numpy as np
import sleap_io as sio
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from sleap_nn.architectures.unet3d import (
    UNet3D,
    n_down_blocks,
    soft_argmax,
    volume_ce,
    volume_moments,
)
from sleap_nn.config.data_config import MultiviewConfig, VolumeAugmentationConfig
from sleap_nn.config.training_job_config import verify_training_cfg
from sleap_nn.config.utils import check_output_strides
from sleap_nn.data import multiview as MV
from sleap_nn.data.volume_input import (
    CenteredVolumeDataset,
    FrozenCenteredInstance,
    VolumeInput,
    VolumeSession,
    collate_volume,
    target_centers,
)
from sleap_nn.io.volume3d import measure_cov_scale
from sleap_nn.training.model_trainer import ModelTrainer

# jitter "auto": the study's 15 world units on a 384-unit cube
JITTER_FRACTION = 15.0 / 384.0
VAL_MONITOR = "val/median_error"


def volume_head(config: DictConfig) -> DictConfig:
    """The ``centered_volume.volume`` head config."""
    return config.model_config.head_configs.centered_volume.volume


def multiview_config(config: DictConfig) -> DictConfig:
    """``data_config.multiview`` with defaults filled in."""
    mv = OmegaConf.select(config, "data_config.multiview")
    base = OmegaConf.structured(MultiviewConfig())
    return OmegaConf.merge(base, mv) if mv is not None else base


def volume_augmentation(config: DictConfig) -> DictConfig:
    """``data_config.augmentation_config.volume`` with defaults filled in."""
    aug = OmegaConf.select(config, "data_config.augmentation_config.volume")
    base = OmegaConf.structured(VolumeAugmentationConfig())
    return OmegaConf.merge(base, aug) if aug is not None else base


def default_passes(center_nodes: Sequence[str], anchor_part: Optional[str]) -> int:
    """Volume passes by default: 1 when the cube centre is the 2D model's anchor, else 2.

    The study's rule. The second pass re-centres the cube (and the crops) on the first
    pass's centre. It helped whenever the cube was centred on other nodes than the 2D
    model's anchor (SLAP -10 %, Ivan's rig -3 %, MultiCam_18 -1 to -9 %, the May 6 hand
    labels -3 to -7 %), and did nothing or lost when they were the same node
    (DoublePoke's Trunk: the anchor's map only repeats the crop centre).
    """
    return 1 if list(center_nodes) == [anchor_part] else 2


# ---------------------------------------------------------------------------------------
# Running the model
# ---------------------------------------------------------------------------------------


def run_volume(
    net: torch.nn.Module,
    builder: VolumeInput,
    batch: Dict[str, torch.Tensor],
    centers: torch.Tensor,
    view_mask: torch.Tensor,
    net_dtype: Optional[torch.dtype] = None,
    covariance: bool = True,
) -> Dict[str, torch.Tensor]:
    """One pass of the volume on a batch of animals (no augmentation, yaw 0).

    Args:
        net: The 3D UNet.
        builder: The input builder (2D model, channels, cube).
        batch: Collated items (``images``, ``image_hw``, cameras, ``up``).
        centers: ``(B, 3)`` cube centres.
        view_mask: ``(B, V)`` cameras in use.
        net_dtype: Autocast dtype of the 3D network on CUDA (``None``: float32).
        covariance: Also compute each point's covariance.

    Returns:
        ``means`` (B, K, 3) float32 (the soft-argmax), ``peak`` (B, K) the largest
        voxel probability, and ``covs`` (B, K, 3, 3) float64 when asked.
    """
    up = batch["up"]
    B = up.shape[0]
    basis = MV.cube_basis(up, torch.zeros(B, device=up.device, dtype=up.dtype))
    cams = {k: batch[k] for k in ("K", "dist", "R", "t")}
    inp, grid = builder(
        batch["images"], cams, centers, basis, view_mask, batch.get("image_hw")
    )
    use_amp = inp.is_cuda and net_dtype is not None
    with torch.autocast(
        device_type=inp.device.type, dtype=net_dtype or torch.float32, enabled=use_amp
    ):
        if inp.is_cuda:
            inp = inp.contiguous(memory_format=torch.channels_last_3d)
        logits = net(inp)
    coords, p = soft_argmax(logits.flatten(2), grid)
    out = {"means": coords, "peak": p.max(-1).values}
    if covariance:
        _, cov = volume_moments(p, builder.n_voxels, builder.cube_size, centers, basis)
        out["covs"] = cov
    return out


def centers_from_points(
    X: torch.Tensor, part_names: Sequence[str], center_nodes: Sequence[str]
) -> torch.Tensor:
    """``(B, 3)`` cube centres: the mean of ``center_nodes`` in ``(B, K, 3)`` points."""
    idx = [list(part_names).index(n) for n in center_nodes]
    return X[:, idx].mean(1)


# ---------------------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------------------


class CenteredVolumeLightningModule(L.LightningModule):
    """Trains the 3D UNet on unprojected maps of a frozen centered-instance model.

    Only the 3D network is a submodule, so checkpoints hold it alone (about 23 MB at
    the defaults); the 2D model is referenced by folder and sha256 in the config.

    Args:
        config: The training job config (``centered_volume`` head, ``unet3d``
            backbone), with ``part_names``, ``center_nodes`` and ``cube_size`` resolved.
        ci: The frozen 2D model (loaded from the config when ``None``).
        steps_per_epoch: Optimizer steps per epoch (for the per-step scheduler).
    """

    def __init__(
        self,
        config: DictConfig,
        ci: Optional[FrozenCenteredInstance] = None,
        steps_per_epoch: Optional[int] = None,
    ):
        """Build the network and the input builder."""
        super().__init__()
        self.config = config
        vol = volume_head(config)
        bb = config.model_config.backbone_config.unet3d
        self.part_names = list(vol.part_names)
        self.center_nodes = list(vol.center_nodes)
        K = len(self.part_names)
        self.net = UNet3D(
            in_ch=K,
            out_ch=K,
            filters=int(bb.filters),
            rate=float(bb.filters_rate),
            down=n_down_blocks(int(bb.max_stride)),
            norm=str(bb.norm),
            convs=int(bb.convs_per_block),
        )
        if ci is None:
            ci = FrozenCenteredInstance.load(
                vol.centered_instance_model, sha256=vol.centered_instance_sha256
            )
        self.builder = VolumeInput(
            ci=ci,
            part_names=self.part_names,
            n_voxels=int(vol.n_voxels),
            cube_size=float(vol.cube_size),
            image_decode=str(vol.image_decode),
        )
        self.n = int(vol.n_voxels)
        self.size = float(vol.cube_size)
        self.sigma = float(vol.ce_sigma_voxels) * self.size / self.n
        self.ce_weight = float(vol.ce_weight)
        self.aug = volume_augmentation(config)
        jit = self.aug.jitter
        self.jitter = self.size * JITTER_FRACTION if jit == "auto" else float(jit)
        self.use_augmentations = bool(
            OmegaConf.select(
                config, "data_config.use_augmentations_train", default=True
            )
        )
        self.steps_per_epoch = steps_per_epoch
        self._val: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self._val_loss: List[float] = []

    # -- devices --------------------------------------------------------------------

    def _ci_to_device(self):
        self.builder.ci.to(self.device)

    def on_fit_start(self):
        """Move the frozen 2D model with the network."""
        self._ci_to_device()
        if self.device.type == "cuda":
            self.net.to(memory_format=torch.channels_last_3d)

    def on_validation_start(self):
        """Move the frozen 2D model with the network."""
        self._ci_to_device()

    # -- input and loss ---------------------------------------------------------------

    def _view_mask(self, batch, train: bool) -> torch.Tensor:
        ok = batch["view_ok"]
        B, V = ok.shape
        vmask = ok.clone()
        if train and self.use_augmentations and self.aug.view_dropout > 0:
            drop = torch.rand(B, V, device=ok.device) < self.aug.view_dropout
            vmask = ok & ~drop
            for b in range(B):
                if vmask[b].sum() < self.aug.min_views:
                    avail = torch.nonzero(ok[b]).flatten()
                    keep = avail[torch.randperm(len(avail), device=ok.device)]
                    vmask[b] = False
                    vmask[b, keep[: self.aug.min_views]] = True
        return vmask.float()

    def build_input(self, batch, train: bool):
        """Augment (in training), crop, read the 2D maps and unproject them."""
        center = batch["center"]
        up = batch["up"]
        B = center.shape[0]
        dev = center.device
        aug = train and self.use_augmentations
        if aug and self.jitter > 0:
            center = center + (torch.rand(B, 3, device=dev) * 2 - 1) * self.jitter
        if aug and self.aug.tilt_degrees > 0:
            up = _tilt(up, math.radians(self.aug.tilt_degrees))
        yaw = (
            torch.rand(B, device=dev) * 2 * math.pi
            if (aug and self.aug.random_yaw)
            else torch.zeros(B, device=dev)
        )
        basis = MV.cube_basis(up, yaw)
        vmask = self._view_mask(batch, train)
        images = batch["images"]
        cams = {k: batch[k] for k in ("K", "dist", "R", "t")}
        hw = batch.get("image_hw")
        if aug and self.aug.max_views and images.shape[1] > self.aug.max_views:
            sel = torch.randperm(images.shape[1], device=dev)[: self.aug.max_views]
            sel = sel.sort().values
            images = images[:, sel]
            cams = {k: v[:, sel] for k, v in cams.items()}
            vmask = vmask[:, sel]
            hw = hw[:, sel] if hw is not None else None
        photo = None
        if aug and self.aug.photometric > 0:
            V = images.shape[1]
            a = 1 + (torch.rand(B, V, device=dev) * 2 - 1) * self.aug.photometric
            b = (torch.rand(B, V, device=dev) * 2 - 1) * self.aug.photometric * 0.5
            photo = (a, b)
        inp, grid = self.builder(images, cams, center, basis, vmask, hw, photo)
        return inp, grid, basis, center

    def _loss(self, batch, train: bool):
        inp, grid, basis, cen = self.build_input(batch, train)
        tgt = batch["X"]
        local = torch.einsum(
            "bkc,bdc->bkd", torch.nan_to_num(tgt) - cen[:, None], basis
        )
        inside = (local.abs() < self.size / 2 - self.size / self.n / 2).all(-1)
        valid = torch.isfinite(tgt).all(-1) & inside
        with torch.autocast(
            device_type=inp.device.type, dtype=torch.bfloat16, enabled=inp.is_cuda
        ):
            if inp.is_cuda:
                inp = inp.contiguous(memory_format=torch.channels_last_3d)
            logits = self.net(inp)
        coords, p = soft_argmax(logits.flatten(2), grid)
        nvalid = valid.sum().clamp(min=1)
        l1 = ((coords - torch.nan_to_num(tgt)).abs().sum(-1) * valid).sum() / nvalid
        loss = l1
        if self.ce_weight > 0:
            loss = loss + self.ce_weight * volume_ce(
                p, grid, tgt, valid.float(), self.sigma
            )
        return loss, l1, coords

    def training_step(self, batch, batch_idx):
        """One optimizer step's loss."""
        loss, l1, _ = self._loss(batch, train=True)
        B = batch["center"].shape[0]
        self.log(
            "train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=B
        )
        self.log("train/l1_per_axis", l1 / 3, on_step=True, on_epoch=True, batch_size=B)
        opt = self.optimizers(use_pl_optimizer=False)
        if opt is not None and opt.param_groups:
            self.log(
                "learning_rate", opt.param_groups[0]["lr"], on_step=True, on_epoch=False
            )
        return loss

    def validation_step(self, batch, batch_idx):
        """Errors of every target point (world units)."""
        loss, _, coords = self._loss(batch, train=False)
        tgt = batch["X"]
        err = (coords - tgt).norm(dim=-1)
        self._val.append((err.detach().cpu(), batch["reliable"].cpu()))
        self._val_loss.append(float(loss))

    def on_validation_epoch_end(self):
        """Log the study's validation numbers: median, mean and 90th percentile error."""
        if not self._val:
            return
        e = torch.cat([v[0] for v in self._val]).numpy()
        r = torch.cat([v[1] for v in self._val]).numpy().astype(bool)
        self._val, losses = [], self._val_loss
        self._val_loss = []
        fin = np.isfinite(e)
        er = e[fin & r] if (fin & r).any() else e[fin]
        if er.size == 0:
            return
        self.log("val/loss", float(np.mean(losses)), sync_dist=True)
        self.log(VAL_MONITOR, float(np.median(er)), prog_bar=True, sync_dist=True)
        self.log("val/mean_error", float(er.mean()), sync_dist=True)
        self.log("val/p90_error", float(np.percentile(er, 90)), sync_dist=True)
        per_node = []
        for k, name in enumerate(self.part_names):
            m = fin[:, k] & r[:, k]
            if m.any():
                per_node.append(f"{name}:{np.median(e[m, k]):.2f}")
        logger.info(
            f"Validation (epoch {self.current_epoch}): reliable median "
            f"{np.median(er):.3f}, mean {er.mean():.3f}, p90 "
            f"{np.percentile(er, 90):.3f}; per node median {' '.join(per_node)}"
        )

    # -- optimizer --------------------------------------------------------------------

    def configure_optimizers(self):
        """AdamW; warmup + cosine stepped every step (the study's), or plateau."""
        tc = self.config.trainer_config
        vol = volume_head(self.config)
        lr = float(tc.optimizer.lr)
        name = str(tc.optimizer_name).lower()
        opt_cls = torch.optim.Adam if name == "adam" else torch.optim.AdamW
        opt = opt_cls(
            self.net.parameters(),
            lr=lr,
            weight_decay=float(vol.weight_decay),
            amsgrad=bool(tc.optimizer.amsgrad),
        )
        sch = OmegaConf.select(tc, "lr_scheduler.cosine_annealing_warmup")
        if sch is not None:
            spe = int(self.steps_per_epoch or tc.train_steps_per_epoch or 1)
            warm = max(1, int(sch.warmup_epochs) * spe)
            total = max(1, int(sch.max_epochs or tc.max_epochs) * spe)
            start_r = float(sch.warmup_start_lr) / lr
            eta_r = float(sch.eta_min) / lr

            def lam(s):
                w = start_r + (1 - start_r) * min(1.0, (s + 1) / warm)
                c = 0.5 * (1 + math.cos(math.pi * min(1.0, s / total)))
                return w * (eta_r + (1 - eta_r) * c)

            return {
                "optimizer": opt,
                "lr_scheduler": {
                    "scheduler": torch.optim.lr_scheduler.LambdaLR(opt, lam),
                    "interval": "step",
                },
            }
        plateau = OmegaConf.select(tc, "lr_scheduler.reduce_lr_on_plateau")
        if plateau is not None:
            min_lr = plateau.min_lr
            return {
                "optimizer": opt,
                "lr_scheduler": {
                    "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                        opt,
                        mode="min",
                        threshold=plateau.threshold,
                        threshold_mode=plateau.threshold_mode,
                        cooldown=plateau.cooldown,
                        patience=plateau.patience,
                        factor=plateau.factor,
                        min_lr=float(min_lr) if min_lr is not None else 0.0,
                    ),
                    "monitor": VAL_MONITOR,
                    "interval": "epoch",
                },
            }
        return opt

    def on_save_checkpoint(self, checkpoint):
        """Record what the network needs besides its weights."""
        vol = volume_head(self.config)
        checkpoint["centered_volume"] = dict(
            part_names=self.part_names,
            center_nodes=self.center_nodes,
            n_voxels=self.n,
            cube_size=self.size,
            centered_instance_model=self.builder.ci.path,
            centered_instance_sha256=self.builder.ci.sha256,
            image_decode=str(vol.image_decode),
        )


def _tilt(up: torch.Tensor, max_rad: float) -> torch.Tensor:
    """Tilt unit vectors by up to ``max_rad`` in a random direction."""
    B = up.shape[0]
    rnd = torch.randn(B, 3, device=up.device, dtype=up.dtype)
    axis = rnd - (rnd * up).sum(-1, keepdim=True) * up
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    ang = torch.rand(B, 1, device=up.device, dtype=up.dtype) * max_rad
    out = torch.cos(ang) * up + torch.sin(ang) * axis
    return out / out.norm(dim=-1, keepdim=True)


# ---------------------------------------------------------------------------------------
# Loading a trained volume
# ---------------------------------------------------------------------------------------


def build_net(config: DictConfig) -> UNet3D:
    """The 3D UNet of a ``centered_volume`` config (untrained)."""
    vol = volume_head(config)
    bb = config.model_config.backbone_config.unet3d
    K = len(vol.part_names)
    return UNet3D(
        in_ch=K,
        out_ch=K,
        filters=int(bb.filters),
        rate=float(bb.filters_rate),
        down=n_down_blocks(int(bb.max_stride)),
        norm=str(bb.norm),
        convs=int(bb.convs_per_block),
    )


def load_volume_model(model_dir: str, device="cpu") -> Tuple[UNet3D, DictConfig]:
    """A trained volume: its 3D UNet (eval mode) and its training config.

    Args:
        model_dir: The run folder (``training_config.yaml`` + ``best.ckpt``).
        device: Torch device.

    Returns:
        A tuple ``(net, config)``.
    """
    from sleap_nn.inference.loaders import _load_training_config

    config, _ = _load_training_config(model_dir)
    net = build_net(config)
    ckpt = torch.load(
        os.path.join(model_dir, "best.ckpt"), map_location="cpu", weights_only=False
    )
    state = {
        k[len("net.") :]: v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("net.")
    }
    net.load_state_dict(state)
    return net.eval().to(device), config


def load_study_checkpoint(path: str, device="cpu") -> Tuple[UNet3D, Dict]:
    """A 2026-11 study checkpoint (``last.pt`` / ``best.pt``) as a packaged ``UNet3D``.

    The study saved ``{"state": state_dict, "args": {...}, "step": n}``. Its module names
    and skip order are this package's, so the state dict loads unchanged.

    Returns:
        A tuple ``(net, args)``.
    """
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = ck["args"]
    K = ck["state"]["head.weight"].shape[0]
    in_ch = ck["state"]["enc.0.0.weight"].shape[1]
    net = UNet3D(
        in_ch, K, int(a["filters"]), float(a["rate"]), int(a["down"]), a["norm"]
    )
    net.load_state_dict(ck["state"])
    return net.eval().to(device), a


# ---------------------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------------------


def _time_block_split(data: MV.SessionData, rows: np.ndarray, fraction: float):
    """Split rows by time: the last ``fraction`` of the session's frames validate."""
    if fraction <= 0 or len(rows) == 0:
        return rows, rows[:0]
    frames = np.unique(data.frame_idx[rows])
    k = int(round(len(frames) * (1 - fraction)))
    k = min(max(k, 1), len(frames)) if len(frames) > 1 else len(frames)
    if k >= len(frames):
        return rows, rows[:0]
    cut = frames[k]
    is_val = data.frame_idx[rows] >= cut
    return rows[~is_val], rows[is_val]


def _auto_cube_size(
    sessions: List[VolumeSession], rows: List[np.ndarray], n: int
) -> float:
    """``cube_size: auto``: twice the largest per-node 99th-percentile distance from the centre.

    Rounded up to a voxel of a tenth of a world unit. This is not the study's rule: on
    its training samples it gives SLAP 412.8 (the study used 384) and DoublePoke 268.8
    (249.6), the tail tip setting it every time. The Euclidean reach also overstates
    what the loss needs (its inside test is per axis in the yawed floor frame).
    """
    d_all = []
    for S, r in zip(sessions, rows):
        if len(r) == 0:
            continue
        d = np.linalg.norm(S.data.X[r] - S.centers[r][:, None], axis=-1)
        d_all.append(np.where(S.targets[r], d, np.nan))
    if not d_all:
        raise ValueError("No training targets to size the cube from.")
    d = np.concatenate(d_all)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        p99 = np.nanpercentile(d, 99, axis=0)
    reach = float(np.nanmax(p99))
    voxel = math.ceil(2 * reach / n / 0.1 - 1e-9) * 0.1
    return round(voxel * n, 6)


@attrs.define
class VolumeModelTrainer(ModelTrainer):
    """``ModelTrainer`` for the volumetric model: multi-camera data, the volume module.

    It reuses the 2D trainer's seed, run-folder and accelerator handling and replaces
    its labels, config, dataset and loop steps, which assume one camera.
    """

    sessions: List[VolumeSession] = attrs.field(factory=list)
    train_rows: List[np.ndarray] = attrs.field(factory=list)
    val_rows: List[np.ndarray] = attrs.field(factory=list)
    ci: Optional[FrozenCenteredInstance] = None

    def _initialize_from_config(self, train_labels=None, val_labels=None):
        """Load labels and resolve the volume's config from them."""
        self.config = verify_training_cfg(self.config)
        self.model_type = "centered_volume"
        self.backbone_type = "unet3d"
        if self.config.trainer_config.seed is not None:
            self._set_seed()
        dc = self.config.data_config
        if train_labels is None and val_labels is None:
            train_labels = [sio.load_slp(p) for p in dc.train_labels_path]
            val_labels = (
                [sio.load_slp(p) for p in dc.val_labels_path]
                if dc.val_labels_path
                else None
            )
        self.train_labels = list(train_labels or [])
        self.val_labels = list(val_labels or [])
        if self._initial_config is None:
            self._initial_config = self.config.copy()
        self.setup_config()

    def setup_config(self):
        """Resolve parts, centre nodes, floor normals, the split and the cube size."""
        cfg = self.config
        vol = volume_head(cfg)
        mv = multiview_config(cfg)
        cfg.data_config.multiview = mv
        if cfg.data_config.augmentation_config is None:
            cfg.data_config.augmentation_config = {}
        cfg.data_config.augmentation_config.volume = volume_augmentation(cfg)
        if self.ci is None:
            self.ci = FrozenCenteredInstance.load(vol.centered_instance_model)
        ci = self.ci
        vol.centered_instance_model = ci.path
        vol.centered_instance_sha256 = ci.sha256

        tr_sessions = [s for lb in self.train_labels for s in lb.sessions]
        va_sessions = [s for lb in self.val_labels for s in lb.sessions]
        if not tr_sessions:
            raise ValueError(
                "centered_volume trains on .slp files with recording sessions (cameras, "
                "videos and instance groups); the training labels have none."
            )
        all_sessions = tr_sessions + va_sessions

        # part names: given, else the 2D model's parts that have 3D targets
        if vol.part_names is None:
            probe = [
                MV.extract_session_data(s, ci.part_names, mv.target_source)
                for s in all_sessions
            ]
            has = np.zeros(len(ci.part_names), bool)
            for d in probe:
                if len(d):
                    has |= d.target_mask(mv.min_views, mv.max_reproj_px).any(0)
            vol.part_names = [n for n, h in zip(ci.part_names, has) if h]
        part_names = list(vol.part_names)
        ci.channel_index(part_names)  # every part must have a 2D map
        if vol.center_nodes is None:
            if ci.anchor_part is None or ci.anchor_part not in part_names:
                raise ValueError(
                    "centered_volume.volume.center_nodes must be given: the 2D model's "
                    f"anchor ({ci.anchor_part!r}) is not among part_names, so the cube "
                    "centre cannot default to it. Name the nodes whose mean is where the "
                    "2D model centres its crops (SLAP's added Center: [Neck, TTI])."
                )
            vol.center_nodes = [ci.anchor_part]
        elif list(vol.center_nodes) != [ci.anchor_part]:
            logger.warning(
                f"The cube is centred on {list(vol.center_nodes)}, the 2D model's crops on "
                f"{ci.anchor_part!r}. They must be the same point (an added midpoint "
                "anchor with these nodes is fine); otherwise every node degrades."
            )
        missing = [n for n in vol.center_nodes if n not in part_names]
        if missing:
            raise ValueError(
                f"center_nodes {missing} are not in part_names {part_names}."
            )

        # sessions, floor normals, rows
        self.sessions, self.train_rows, self.val_rows = [], [], []
        n_val_sessions = len(va_sessions)
        for i, s in enumerate(all_sessions):
            d = MV.extract_session_data(s, part_names, mv.target_source)
            targets = d.target_mask(mv.min_views, mv.max_reproj_px)
            reliable = d.reliable()
            fit = (
                MV.resolve_up(
                    s,
                    d.cameras,
                    d.dlt,
                    MV.reliable_mask(d.dlt, d.nview, d.err),
                    part_names,
                    up=list(mv.up) if mv.up is not None else None,
                    up_fit_nodes=list(mv.up_fit_nodes) if mv.up_fit_nodes else None,
                    band=float(mv.floor_band),
                )
                if len(d)
                else MV.FloorFit(up=np.array([0.0, 0.0, 1.0]), method="none")
            )
            cen = (
                target_centers(d.X, targets, part_names, vol.center_nodes)
                if len(d)
                else np.zeros((0, 3))
            )
            ok = (targets.sum(-1) >= mv.min_nodes) & np.isfinite(cen).all(-1)
            rows = np.nonzero(ok)[0]
            S = VolumeSession(
                data=d,
                up=fit.up,
                targets=targets,
                reliable=reliable,
                centers=cen,
                rows=rows,
            )
            self.sessions.append(S)
            if i >= len(tr_sessions):
                self.train_rows.append(rows[:0])
                self.val_rows.append(rows)
            elif n_val_sessions:
                self.train_rows.append(rows)
                self.val_rows.append(rows[:0])
            else:
                tr, va = _time_block_split(
                    d, rows, float(cfg.data_config.validation_fraction)
                )
                self.train_rows.append(tr)
                self.val_rows.append(va)
            logger.info(
                f"Session {i}: {len(s.camera_group.cameras)} cameras, {len(d)} animal-frames, "
                f"{len(rows)} usable; up {np.round(fit.up, 3).tolist()} ({fit.method})"
            )
        n_tr = sum(len(r) for r in self.train_rows)
        n_va = sum(len(r) for r in self.val_rows)
        if n_tr == 0:
            raise ValueError(
                "No training samples: no instance group has enough 3D target nodes "
                f"(min_nodes {mv.min_nodes}) and a cube centre ({list(vol.center_nodes)})."
            )
        logger.info(
            f"centered_volume: {n_tr} training and {n_va} validation animal-frames"
        )

        if vol.cube_size == "auto":
            vol.cube_size = _auto_cube_size(
                self.sessions, self.train_rows, int(vol.n_voxels)
            )
            logger.info(f"cube_size auto: {vol.cube_size} world units")
        cfg.model_config.backbone_config.unet3d.in_channels = len(part_names)
        self.config = check_output_strides(cfg)
        self._verify_accelerator_config()
        if self.config.trainer_config.trainer_devices is None:
            self.config.trainer_config.trainer_devices = "auto"
        self._setup_ckpt_path()

    def _run_dir(self) -> Path:
        tc = self.config.trainer_config
        return Path(tc.ckpt_dir) / tc.run_name

    def _dataset(
        self, rows: List[np.ndarray], limit: Optional[int] = None, seed: int = 0
    ):
        samples = np.array(
            [(si, int(r)) for si, rr in enumerate(rows) for r in rr], np.int64
        ).reshape(-1, 2)
        if limit is not None and len(samples) > limit:
            samples = samples[
                np.random.default_rng(seed).permutation(len(samples))[:limit]
            ]
        mv = self.config.data_config.multiview
        vol = volume_head(self.config)
        return CenteredVolumeDataset(
            self.sessions,
            samples,
            channels=self.ci.in_channels,
            image_decode=str(vol.image_decode),
            input_scale=self.ci.input_scale,
            cache_frames=str(mv.cache_frames),
            jpeg_quality=int(mv.jpeg_quality),
        )

    def _write_split(self):
        split = {"train": [], "val": []}
        for key, rows in (("train", self.train_rows), ("val", self.val_rows)):
            for si, rr in enumerate(rows):
                d = self.sessions[si].data
                split[key].append(
                    dict(
                        session=si,
                        cameras=d.cameras.names,
                        frames=sorted({int(f) for f in d.frame_idx[rr]}),
                        n_samples=int(len(rr)),
                    )
                )
        with open(self._run_dir() / "split.json", "w") as f:
            json.dump(split, f, indent=1)

    @staticmethod
    def _workers(value) -> int:
        if value == "auto":
            n = (
                len(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity")
                else os.cpu_count()
            )
            return max(0, min(8, (n or 1) - 1))
        return int(value or 0)

    def train(self):
        """Fit the volume, then measure its validation covariance scale."""
        if not self.sessions:
            self._initialize_from_config()
        tc = self.config.trainer_config
        run_dir = self._run_dir()
        rank0 = int(os.environ.get("LOCAL_RANK", "0")) == 0
        if rank0:
            run_dir.mkdir(parents=True, exist_ok=True)
        train_ds = self._dataset(self.train_rows)
        val_rows = self.val_rows
        if sum(len(r) for r in val_rows) == 0:
            logger.warning(
                "No validation samples (a single short session?): validating on the "
                "training samples."
            )
            val_rows = self.train_rows
        mv = self.config.data_config.multiview
        val_ds = self._dataset(
            val_rows, limit=int(mv.val_samples) if mv.val_samples else None
        )
        bs = int(tc.train_data_loader.batch_size)
        spe = tc.train_steps_per_epoch
        if spe is None:
            spe = max(int(tc.min_train_steps_per_epoch), math.ceil(len(train_ds) / bs))
        tc.train_steps_per_epoch = int(spe)
        self.lightning_model = CenteredVolumeLightningModule(
            self.config, ci=self.ci, steps_per_epoch=int(spe)
        )
        n_params = sum(p.numel() for p in self.lightning_model.net.parameters())
        self.config.model_config.total_params = n_params
        vol = volume_head(self.config)
        logger.info(
            f"UNet3D channels {self.lightning_model.net.channels}, {n_params / 1e6:.2f} M "
            f"parameters; cube {vol.cube_size} / {vol.n_voxels} = "
            f"{float(vol.cube_size) / int(vol.n_voxels):.3f} units per voxel"
        )
        callbacks = []
        monitor = (
            tc.model_ckpt.monitor
            if tc.model_ckpt.monitor != "val/loss"
            else VAL_MONITOR
        )
        if tc.save_ckpt:
            from lightning.pytorch.callbacks import ModelCheckpoint

            callbacks.append(
                ModelCheckpoint(
                    dirpath=str(run_dir),
                    filename="best",
                    monitor=monitor,
                    mode=tc.model_ckpt.mode,
                    save_top_k=1,
                    save_last=bool(tc.model_ckpt.save_last),
                    auto_insert_metric_name=False,
                    enable_version_counter=False,
                )
            )
        if tc.early_stopping.stop_training_on_plateau:
            from lightning.pytorch.callbacks import EarlyStopping

            callbacks.append(
                EarlyStopping(
                    monitor=monitor,
                    mode=tc.model_ckpt.mode,
                    min_delta=tc.early_stopping.min_delta,
                    patience=tc.early_stopping.patience,
                )
            )
        from lightning.pytorch.loggers import CSVLogger

        loggers = [CSVLogger(save_dir=str(run_dir), name="", version="")]
        if tc.use_wandb:
            from lightning.pytorch.loggers import WandbLogger

            loggers.append(
                WandbLogger(
                    project=tc.wandb.project,
                    entity=tc.wandb.entity,
                    name=tc.wandb.name or tc.run_name,
                    save_dir=str(run_dir),
                )
            )
        devices = tc.trainer_device_indices or tc.trainer_devices
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self.trainer = L.Trainer(
            callbacks=callbacks,
            logger=loggers,
            enable_checkpointing=bool(tc.save_ckpt),
            devices=devices,
            max_epochs=int(tc.max_epochs),
            accelerator=tc.trainer_accelerator,
            strategy=tc.trainer_strategy,
            enable_progress_bar=bool(tc.enable_progress_bar),
            gradient_clip_val=vol.grad_clip_norm,
            log_every_n_steps=min(50, int(spe)),
            num_sanity_val_steps=0,
            use_distributed_sampler=False,
        )
        nw = self._workers(tc.train_data_loader.num_workers)
        tc.train_data_loader.num_workers = nw
        # each process draws its own samples (seed + rank), as the study's one-GPU
        # run drew with replacement
        gen = torch.Generator().manual_seed(
            int(tc.seed or 0) + int(self.trainer.global_rank)
        )
        sampler = torch.utils.data.RandomSampler(
            train_ds, replacement=True, num_samples=int(spe) * bs, generator=gen
        )
        train_dl = torch.utils.data.DataLoader(
            train_ds,
            batch_size=bs,
            sampler=sampler,
            num_workers=nw,
            collate_fn=collate_volume,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=nw > 0,
            drop_last=True,
        )
        vnw = self._workers(tc.val_data_loader.num_workers)
        tc.val_data_loader.num_workers = vnw
        val_dl = torch.utils.data.DataLoader(
            val_ds,
            batch_size=int(tc.val_data_loader.batch_size or bs),
            shuffle=False,
            num_workers=vnw,
            collate_fn=collate_volume,
            persistent_workers=vnw > 0,
        )
        if self.trainer.global_rank == 0:
            self._initial_config.sleap_nn_version = _version()
            OmegaConf.save(
                self._initial_config, (run_dir / "initial_config.yaml").as_posix()
            )
            self._write_split()
            self._save_config()
        self.trainer.fit(
            self.lightning_model, train_dl, val_dl, ckpt_path=tc.resume_ckpt_path
        )
        if self.trainer.global_rank == 0:
            best = run_dir / "best.ckpt"
            if tc.save_ckpt and best.exists():
                ck = torch.load(best, map_location="cpu", weights_only=False)
                self.lightning_model.load_state_dict(ck["state_dict"])
            try:
                scale, n = self.validation_cov_scale(val_rows)
            except Exception as e:  # noqa: BLE001 - a fallback value only
                logger.warning(f"Could not measure validation_cov_scale: {e}")
                scale, n = None, 0
            if scale is not None:
                vol.validation_cov_scale = float(scale)
                logger.info(f"validation_cov_scale {scale:.4f} from {n} points")
            else:
                logger.warning(
                    f"validation_cov_scale not measured: {n} validation points with >= 4 "
                    "views and < 3 px (20 needed). Inference measures it on each run."
                )
            self._save_config()

    def _save_config(self):
        self.config.sleap_nn_version = _version()
        OmegaConf.save(
            self.config, (self._run_dir() / "training_config.yaml").as_posix()
        )

    @torch.no_grad()
    def validation_cov_scale(
        self, rows: List[np.ndarray]
    ) -> Tuple[Optional[float], int]:
        """``cov_scale`` on validation animal-frames, with cubes centred as at inference.

        Cubes are centred on the robust DLT of each group's 2D points (not the target),
        with the default number of passes; the scale is LUC3D's exporter's measure
        against that DLT (see :func:`sleap_nn.io.volume3d.measure_cov_scale`).
        """
        module = self.lightning_model
        device = module.device if module.device.type != "meta" else torch.device("cpu")
        module.eval()
        module.builder.ci.to(device)
        vol = volume_head(self.config)
        part_names = list(vol.part_names)
        passes = default_passes(vol.center_nodes, self.ci.anchor_part)
        ds = self._dataset(rows, limit=self.config.data_config.multiview.val_samples)
        dl = torch.utils.data.DataLoader(
            ds,
            batch_size=int(self.config.trainer_config.train_data_loader.batch_size),
            shuffle=False,
            collate_fn=collate_volume,
        )
        means, covs, X, nv, er = [], [], [], [], []
        for batch in dl:
            idx = [
                (int(s), int(f), int(g))
                for s, f, g in zip(
                    batch["session"], batch["frame_idx"], batch["group_index"]
                )
            ]
            dlt, nview, err = [], [], []
            for s, f, g in idx:
                d = self.sessions[s].data
                r = np.nonzero((d.frame_idx == f) & (d.group_index == g))[0][0]
                dlt.append(d.dlt[r])
                nview.append(d.nview[r])
                err.append(d.err[r])
            dlt = np.stack(dlt)
            cen = MV.cube_centers(dlt, part_names, vol.center_nodes)
            ok = np.isfinite(cen).all(-1)
            if not ok.any():
                continue
            batch = {
                k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
            }
            centers = torch.as_tensor(
                np.nan_to_num(cen), dtype=torch.float32, device=device
            )
            vmask = batch["view_ok"].float()
            out = None
            for _ in range(passes):
                out = run_volume(
                    module.net, module.builder, batch, centers, vmask, torch.bfloat16
                )
                centers = centers_from_points(
                    out["means"], part_names, vol.center_nodes
                )
            m = out["means"].double().cpu().numpy()
            m[~ok] = np.nan
            means.append(m)
            covs.append(out["covs"].cpu().numpy())
            X.append(dlt)
            nv.append(np.stack(nview))
            er.append(np.stack(err))
        if not means:
            return None, 0
        return measure_cov_scale(
            np.concatenate(means),
            np.concatenate(covs),
            np.concatenate(X),
            np.concatenate(nv),
            np.concatenate(er),
        )


def _version() -> str:
    import sleap_nn

    return sleap_nn.__version__


def train_volume(
    config: DictConfig, train_labels=None, val_labels=None
) -> VolumeModelTrainer:
    """Train a ``centered_volume`` model from a config (``sleap-nn train``'s entry)."""
    trainer = VolumeModelTrainer(config=config)
    trainer._initialize_from_config(train_labels=train_labels, val_labels=val_labels)
    trainer.train()
    return trainer
