"""Tests for the volumetric model's 3D network, readout and loss (`unet3d`)."""

import math

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from sleap_nn.architectures.unet3d import (
    UNet3D,
    n_down_blocks,
    soft_argmax,
    volume_ce,
    volume_moments,
)
from sleap_nn.data import multiview as MV
from tests.fixtures.volume_study import StudyUNet3D


def test_study_parameter_count():
    """The study's 15-part SLAP volume: 5.85 M parameters, 23.4 MB in float32."""
    net = UNet3D(15, 15)
    n = sum(p.numel() for p in net.parameters())
    assert net.channels == [32, 64, 128, 256]
    assert abs(n - 5.85e6) < 0.01e6
    assert abs(n * 4 / 1e6 - 23.4) < 0.1


def test_study_state_dict_loads_and_matches():
    torch.manual_seed(0)
    study = StudyUNet3D(4, 3, filters=8, rate=2.0, down=3).eval()
    net = UNet3D(4, 3, filters=8, rate=2.0, down=3).eval()
    net.load_state_dict(study.state_dict(), strict=True)
    x = torch.randn(2, 4, 16, 16, 16)
    with torch.no_grad():
        assert torch.equal(net(x), study(x))


def test_skip_order_is_pinned():
    """Swapping the concatenation order changes the output: the test above has teeth."""
    torch.manual_seed(0)
    study = StudyUNet3D(2, 2, filters=4, rate=2.0, down=2).eval()
    net = UNet3D(2, 2, filters=4, rate=2.0, down=2).eval()
    net.load_state_dict(study.state_dict())

    def swapped(x):
        skips = []
        for blk in net.enc:
            x = blk(x)
            skips.append(x)
            x = F.max_pool3d(x, 2)
        x = net.mid(x)
        for blk, s in zip(net.dec, reversed(skips)):
            x = F.interpolate(
                x, size=s.shape[-3:], mode="trilinear", align_corners=False
            )
            x = blk(torch.cat([s, x], 1))  # sleap-nn's 2D decoder order
        return net.head(x)

    x = torch.randn(1, 2, 8, 8, 8)
    with torch.no_grad():
        assert not torch.allclose(swapped(x), net(x))


def test_n_down_blocks():
    assert n_down_blocks(8) == 3 and n_down_blocks(2) == 1
    with pytest.raises(ValueError):
        n_down_blocks(6)
    with pytest.raises(ValueError):
        n_down_blocks(1)


def test_soft_argmax_sum_equals_einsum():
    torch.manual_seed(1)
    logits = torch.randn(2, 3, 512) * 3
    grid = torch.randn(2, 512, 3) * 50
    coords, p = soft_argmax(logits, grid)
    ref = torch.einsum("bkm,bmc->bkc", logits.softmax(-1), grid)
    assert torch.allclose(coords, ref, atol=1e-4)
    assert torch.allclose(p.sum(-1), torch.ones(2, 3))


def test_volume_moments_match_direct_moments():
    """Mean and covariance from the marginals equal the moments over the world grid."""
    n, size = 12, 60.0
    up = torch.tensor([[0.1, -0.2, 1.0]])
    up = up / up.norm()
    basis = MV.cube_basis(up, torch.tensor([0.7]))
    center = torch.tensor([[5.0, -3.0, 20.0]])
    grid = MV.cube_world(center, basis, MV.cube_offsets(n, size)).double()
    # an anisotropic, rotated Gaussian in world coordinates, discretised on the cube
    mu = center[0].double() + torch.tensor([4.0, -2.0, 3.0], dtype=torch.float64)
    A = torch.tensor(
        [[60.0, 15.0, -5.0], [15.0, 30.0, 8.0], [-5.0, 8.0, 20.0]], dtype=torch.float64
    )
    d = grid[0] - mu
    logp = -0.5 * (d @ torch.linalg.inv(A) * d).sum(-1)
    p = torch.softmax(logp, 0)[None, None].float()  # (1, 1, M)
    m, C = volume_moments(p, n, size, center, basis)
    pd = p[0, 0].double()
    m_ref = (pd[:, None] * grid[0]).sum(0)
    dd = grid[0] - m_ref
    C_ref = (pd[:, None, None] * dd[:, :, None] * dd[:, None, :]).sum(0)
    assert torch.allclose(m[0, 0], m_ref, atol=1e-4)
    assert torch.allclose(C[0, 0], C_ref, atol=1e-3)
    # and they are near the Gaussian's own (it fits well inside the cube)
    assert torch.allclose(C[0, 0], A, rtol=0.1, atol=2.0)


def test_volume_ce_masks_invalid_targets():
    torch.manual_seed(2)
    grid = torch.randn(1, 64, 3) * 10
    p = torch.softmax(torch.randn(1, 2, 64), -1)
    tgt = torch.tensor([[[1.0, 2.0, 3.0], [float("nan")] * 3]])
    valid = torch.tensor([[1.0, 0.0]])
    ce = volume_ce(p, grid, tgt, valid, 2.0)
    d2 = ((grid[0] - tgt[0, 0]) ** 2).sum(-1)
    q = torch.softmax(-d2 / 8.0, -1)
    ref = -(q * torch.log(p[0, 0])).sum()
    assert torch.isfinite(ce) and torch.allclose(ce, ref)
    assert volume_ce(p, grid, tgt, torch.zeros(1, 2), 2.0) == 0
