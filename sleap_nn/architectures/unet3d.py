"""The volumetric model's 3D network, readout and loss (model type ``centered_volume``).

A port of the 2026-11-11 volumetric study's ``vol3d.py``: sleap-nn's UNet layout
(filters, filters_rate, down blocks = log2(max_stride), a middle block, an interpolating
decoder, two convolutions per block, a 1x1 head) with 3D layers.

The module names (``enc``, ``mid``, ``dec``, ``head``) and the decoder's concatenation
order (``[upsampled, skip]``, the reverse of sleap-nn's 2D decoder) are the study's, so
a study checkpoint's state dict loads unchanged. Do not change either.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm(kind: str, c: int) -> nn.Module:
    if kind == "none":
        return nn.Identity()
    if kind == "group":
        return nn.GroupNorm(min(8, c), c)
    if kind == "instance":
        return nn.InstanceNorm3d(c, affine=True)
    raise ValueError(f"Unknown norm {kind!r}: expected group, instance or none.")


class Block3d(nn.Sequential):
    """``convs`` x (3x3x3 convolution, norm, ReLU)."""

    def __init__(self, cin: int, cout: int, norm: str = "group", convs: int = 2):
        """Build the block."""
        layers = []
        for i in range(convs):
            layers += [
                nn.Conv3d(cin if i == 0 else cout, cout, 3, padding=1),
                _norm(norm, cout),
                nn.ReLU(inplace=True),
            ]
        super().__init__(*layers)


class UNet3D(nn.Module):
    """3D UNet: one logit volume per body part.

    Args:
        in_ch: Input channels (the body parts read from the 2D model).
        out_ch: Output channels (the body parts predicted).
        filters: Filters of the first block.
        rate: Filter growth per block.
        down: Down blocks (``log2(max_stride)``).
        norm: ``group`` (GroupNorm with ``min(8, c)`` groups), ``instance`` or
            ``none``.
        convs: Convolutions per block.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        filters: int = 32,
        rate: float = 2.0,
        down: int = 3,
        norm: str = "group",
        convs: int = 2,
    ):
        """Build the network."""
        super().__init__()
        ch = [int(round(filters * rate**i)) for i in range(down + 1)]
        self.enc = nn.ModuleList()
        c = in_ch
        for i in range(down):
            self.enc.append(Block3d(c, ch[i], norm, convs))
            c = ch[i]
        self.mid = Block3d(c, ch[down], norm, convs)
        self.dec = nn.ModuleList()
        c = ch[down]
        for i in reversed(range(down)):
            self.dec.append(Block3d(c + ch[i], ch[i], norm, convs))
            c = ch[i]
        self.head = nn.Conv3d(c, out_ch, 1)
        self.channels = ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, in_ch, n, n, n)`` to ``(B, out_ch, n, n, n)`` logits."""
        skips = []
        for blk in self.enc:
            x = blk(x)
            skips.append(x)
            x = F.max_pool3d(x, 2)
        x = self.mid(x)
        for blk, s in zip(self.dec, reversed(skips)):
            x = F.interpolate(
                x, size=s.shape[-3:], mode="trilinear", align_corners=False
            )
            # [upsampled, skip]: the study's order (its checkpoints depend on it)
            x = blk(torch.cat([x, s], 1))
        return self.head(x)


def soft_argmax(
    logits: torch.Tensor, grid: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Expected position under each part's softmax over the cube.

    The readout is a broadcast multiply and a sum in float32, never an ``einsum``: on
    ROCm (AMD GPUs) hipBLASLt runs that product as one long accumulation and moved
    points by a median 0.65 mm in the study.

    Args:
        logits: ``(B, K, M)`` flattened logit volumes.
        grid: ``(B, M, 3)`` voxel centres in world coordinates.

    Returns:
        A tuple ``(coords (B, K, 3), probs (B, K, M))``, both float32.
    """
    p = logits.float().softmax(-1)
    return (p.unsqueeze(-1) * grid.float().unsqueeze(1)).sum(2), p


def volume_ce(
    p: torch.Tensor,
    grid: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Cross-entropy between each part's softmax and a Gaussian around its target.

    Args:
        p: ``(B, K, M)`` softmax probabilities.
        grid: ``(B, M, 3)`` voxel centres.
        target: ``(B, K, 3)`` targets (NaN allowed where ``valid`` is 0).
        valid: ``(B, K)`` 1 where the target counts.
        sigma: The Gaussian's sigma in world units.

    Returns:
        The mean cross-entropy over valid targets.
    """
    d2 = ((grid[:, None, :, :] - torch.nan_to_num(target)[:, :, None, :]) ** 2).sum(-1)
    q = torch.softmax(-d2 / (2 * sigma**2), -1)
    ce = -(q * torch.log(p.clamp(min=1e-12))).sum(-1)
    return (ce * valid).sum() / valid.sum().clamp(min=1)


def volume_moments(
    p: torch.Tensor,
    n: int,
    size: float,
    center: torch.Tensor,
    basis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mean and covariance of each part's softmax, in world coordinates.

    The volume-fusion study's marginal method (``vf_gpu.run_vol``): the second moments
    are summed from the three 2D marginals of the probability cube in float64, in the
    cube's own axes, then rotated by the basis.

    Args:
        p: ``(B, K, n^3)`` softmax probabilities (C order).
        n: Voxels per side.
        size: Cube side.
        center: ``(B, 3)`` cube centres.
        basis: ``(B, 3, 3)`` bases (rows ``e1, e2, up``).

    Returns:
        A tuple ``(mean (B, K, 3), cov (B, K, 3, 3))`` in float64.
    """
    dev = p.device
    ax = (torch.arange(n, device=dev, dtype=torch.float64) + 0.5) * size / n - size / 2
    B, K = p.shape[:2]
    P = p.reshape(B, K, n, n, n).double()
    Pij, Pik, Pjk = P.sum(4), P.sum(3), P.sum(2)
    pi, pj, pk = Pij.sum(3), Pij.sum(2), Pik.sum(2)
    mu = torch.stack([pi @ ax, pj @ ax, pk @ ax], -1)
    a2 = ax * ax
    Eii, Ejj, Ekk = pi @ a2, pj @ a2, pk @ a2
    Eij = ((Pij * ax[:, None]).sum(2) * ax).sum(-1)
    Eik = ((Pik * ax[:, None]).sum(2) * ax).sum(-1)
    Ejk = ((Pjk * ax[:, None]).sum(2) * ax).sum(-1)
    E2 = torch.stack(
        [
            torch.stack([Eii, Eij, Eik], -1),
            torch.stack([Eij, Ejj, Ejk], -1),
            torch.stack([Eik, Ejk, Ekk], -1),
        ],
        -2,
    )
    Cl = E2 - mu[..., :, None] * mu[..., None, :]
    Bm = basis.double()
    Cw = torch.einsum("bdi,bkde,bej->bkij", Bm, Cl, Bm)
    mw = center.double()[:, None] + torch.einsum("bkd,bdj->bkj", mu, Bm)
    return mw, Cw


def n_down_blocks(max_stride: int) -> int:
    """Down blocks for a max stride (a power of 2)."""
    down = int(round(math.log2(max_stride)))
    if 2**down != max_stride or down < 1:
        raise ValueError(f"max_stride must be a power of 2 >= 2; got {max_stride}.")
    return down
