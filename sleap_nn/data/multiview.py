"""Multi-camera geometry for the volumetric 3D model type (``centered_volume``).

Everything here is plain geometry on calibrated cameras: no network. It is a port of
the 2026-11-11 volumetric study (``vol3d.py``, ``slap_targets.py`` and
``rig_targets.py``), which trained and scored the model this package now ships.

Conventions:
    - World coordinates are the calibration's own frame and units (millimetres on most
      rigs; SLAP's calibration unit is about 0.62 mm).
    - Cameras follow OpenCV: ``Xc = R Xw + t``; ``x = Xc[:2] / Xc[2]``; 5-term
      distortion ``[k1, k2, p1, p2, k3]``; ``u = fx x_d + cx``. With only ``k1`` set,
      every function here does exactly the study's arithmetic (k1-only model), so its
      targets and features are reproduced to the last bit.
    - Pixel coordinates are sleap-io's: the centre of the top-left pixel is ``(0, 0)``.
    - A cube has ``n^3`` voxel centres at offsets ``(i + 0.5) S / n - S / 2`` along a
      basis whose rows are ``(e1, e2, up)``: ``up`` is the floor normal and ``e1`` is
      turned by a yaw angle about it. Voxels are in C order (axis 0 slowest).

Main pieces:
    - :class:`CameraArrays`: a session's cameras as arrays (``from_session``).
    - Torch geometry for the model: :func:`cube_offsets`, :func:`cube_basis`,
      :func:`cube_world`, :func:`project_points`, :func:`sample_views`.
    - NumPy geometry for targets and inference: :func:`np_project`,
      :func:`np_undistort`, :func:`robust_triangulate`.
    - :func:`associate_frame`: matches each camera's detections to animals when no
      instance groups exist (the study's ``rig_targets.py`` rule).
    - :func:`fit_floor_normal` / :func:`resolve_up`: the floor normal a cube stands on.
    - :func:`extract_session_data`: a session's instance groups as training targets.
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

import attrs
import cv2
import numpy as np
import sleap_io as sio
import torch
import torch.nn.functional as F
from loguru import logger
from scipy.optimize import linear_sum_assignment

# Robust DLT: drop a point's worst view while its reprojection error exceeds this (px)
# and at least 3 views remain (the study's target rule).
DROP_PX = 12.0
# Matching across cameras: a camera pair proposes an animal when its 2-view median
# reprojection error is below PAIR_PX; another camera supports it within GATE_PX
# (scaled by that camera's image width / 1280); an animal needs MIN_CAMS cameras.
PAIR_PX = 12.0
GATE_PX = 60.0
MIN_CAMS = 3
# A target point is "reliable" (used for scoring and for measuring cov_scale) with at
# least RELIABLE_VIEWS views and a mean reprojection error below RELIABLE_PX.
RELIABLE_VIEWS = 3
RELIABLE_PX = 5.0


# ---------------------------------------------------------------------------------------
# Cameras
# ---------------------------------------------------------------------------------------


@attrs.define
class CameraArrays:
    """A session's calibrated cameras as stacked arrays.

    Attributes:
        names: Camera names, in the session's camera order.
        K: Intrinsic matrices, ``(V, 3, 3)`` float64.
        dist: Distortion ``[k1, k2, p1, p2, k3]``, ``(V, 5)`` float64.
        R: World-to-camera rotations, ``(V, 3, 3)`` float64.
        t: World-to-camera translations, ``(V, 3)`` float64.
        size: Image sizes ``(width, height)``, ``(V, 2)`` int64.
    """

    names: List[str]
    K: np.ndarray
    dist: np.ndarray
    R: np.ndarray
    t: np.ndarray
    size: np.ndarray

    @classmethod
    def from_cameras(cls, cameras: Sequence[sio.Camera]) -> "CameraArrays":
        """Stack sleap-io cameras.

        Args:
            cameras: ``sio.Camera`` objects (matrix, dist, rvec, tvec, size).

        Returns:
            The cameras as arrays, in the given order.
        """
        names, K, dist, R, t, size = [], [], [], [], [], []
        for i, cam in enumerate(cameras):
            names.append(cam.name if cam.name else f"cam{i}")
            K.append(np.asarray(cam.matrix, np.float64).reshape(3, 3))
            d = np.zeros(5, np.float64)
            dd = np.asarray(cam.dist, np.float64).ravel()[:5]
            d[: len(dd)] = dd
            dist.append(d)
            rmat, _ = cv2.Rodrigues(np.asarray(cam.rvec, np.float64).reshape(3, 1))
            R.append(rmat)
            t.append(np.asarray(cam.tvec, np.float64).ravel())
            sz = cam.size if cam.size is not None else (0, 0)
            size.append(np.asarray(sz, np.int64).ravel()[:2])
        return cls(
            names=names,
            K=np.stack(K),
            dist=np.stack(dist),
            R=np.stack(R),
            t=np.stack(t),
            size=np.stack(size),
        )

    @classmethod
    def from_session(cls, session: sio.RecordingSession) -> "CameraArrays":
        """Stack the cameras of a recording session, in its camera-group order."""
        return cls.from_cameras(session.camera_group.cameras)

    @property
    def n_views(self) -> int:
        """Number of cameras."""
        return len(self.names)

    @property
    def centers(self) -> np.ndarray:
        """Camera centres in world coordinates, ``(V, 3)``."""
        return -np.einsum("vij,vi->vj", self.R, self.t)

    def subset(self, views: Sequence[int]) -> "CameraArrays":
        """The cameras at the given indices, in that order."""
        idx = list(views)
        return CameraArrays(
            names=[self.names[i] for i in idx],
            K=self.K[idx],
            dist=self.dist[idx],
            R=self.R[idx],
            t=self.t[idx],
            size=self.size[idx],
        )

    def to_torch(self, device=None, dtype=torch.float32) -> Dict[str, torch.Tensor]:
        """The arrays as tensors (``K``, ``dist``, ``R``, ``t``), each with a view axis."""
        return {
            k: torch.as_tensor(getattr(self, k), dtype=dtype, device=device)
            for k in ("K", "dist", "R", "t")
        }


# ---------------------------------------------------------------------------------------
# Torch geometry (the model's input side)
# ---------------------------------------------------------------------------------------


def cube_offsets(n: int, size: float, device=None) -> torch.Tensor:
    """Voxel-centre offsets of a cube in its own axes.

    Args:
        n: Voxels per side.
        size: Cube side in world units.
        device: Torch device.

    Returns:
        ``(n^3, 3)`` float32 offsets ``(i + 0.5) size / n - size / 2``, C order (axis 0
        slowest).
    """
    v = size / n
    a = (torch.arange(n, device=device, dtype=torch.float32) + 0.5) * v - size / 2
    g = torch.stack(torch.meshgrid(a, a, a, indexing="ij"), -1)
    return g.reshape(-1, 3)


def cube_basis(up: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Cube axes from a floor normal and a yaw angle.

    Args:
        up: ``(B, 3)`` unit floor normals.
        yaw: ``(B,)`` yaw angles about ``up`` in radians.

    Returns:
        ``(B, 3, 3)`` orthonormal bases with rows ``e1, e2, up``.
    """
    ref = torch.tensor([1.0, 0.0, 0.0], device=up.device, dtype=up.dtype).expand_as(up)
    alt = torch.tensor([0.0, 1.0, 0.0], device=up.device, dtype=up.dtype).expand_as(up)
    ref = torch.where((ref * up).sum(-1, keepdim=True).abs() > 0.9, alt, ref)
    e1 = ref - (ref * up).sum(-1, keepdim=True) * up
    e1 = e1 / e1.norm(dim=-1, keepdim=True)
    e2 = torch.cross(up, e1, dim=-1)
    c, s = torch.cos(yaw)[:, None], torch.sin(yaw)[:, None]
    r1 = c * e1 + s * e2
    r2 = -s * e1 + c * e2
    return torch.stack([r1, r2, up], 1)


def cube_world(
    center: torch.Tensor, basis: torch.Tensor, offsets: torch.Tensor
) -> torch.Tensor:
    """World coordinates of the voxel centres.

    Args:
        center: ``(B, 3)`` cube centres.
        basis: ``(B, 3, 3)`` bases (rows ``e1, e2, up``).
        offsets: ``(M, 3)`` offsets from :func:`cube_offsets`.

    Returns:
        ``(B, M, 3)`` world coordinates.
    """
    return center[:, None, :] + offsets[None] @ basis


def _distort(x: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
    """Apply OpenCV distortion to normalised coordinates.

    ``x`` is ``(B, V, M, 2)`` and ``dist`` ``(B, V, 5)``. The radial term is written in
    Horner form, so with ``k2 = k3 = p1 = p2 = 0`` it is bitwise the study's
    ``x * (1 + k1 r^2)``, and it stays finite for the huge ``x`` of points next to a
    camera's image plane (an expanded ``k2 r^4`` would overflow to ``inf * 0 = NaN``).
    """
    k1, k2, p1, p2, k3 = (dist[..., i, None, None] for i in range(5))
    r2 = (x * x).sum(-1, keepdim=True)
    radial = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
    xx, yy = x[..., 0:1], x[..., 1:2]
    dx = 2 * p1 * xx * yy + p2 * (r2 + 2 * xx * xx)
    dy = p1 * (r2 + 2 * yy * yy) + 2 * p2 * xx * yy
    return x * radial + torch.cat([dx, dy], -1)


def project_points(
    Xw: torch.Tensor,
    K: torch.Tensor,
    dist: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project world points into every camera.

    Args:
        Xw: ``(B, M, 3)`` world points.
        K: ``(B, V, 3, 3)`` intrinsics.
        dist: ``(B, V, 5)`` distortion ``[k1, k2, p1, p2, k3]``.
        R: ``(B, V, 3, 3)`` rotations.
        t: ``(B, V, 3)`` translations.

    Returns:
        A tuple ``(uv, z)``: ``(B, V, M, 2)`` distorted pixel coordinates and
        ``(B, V, M)`` camera-space depths (a point is in front of a camera when
        ``z > 1e-3``).
    """
    Xc = torch.einsum("bvij,bmj->bvmi", R, Xw) + t[:, :, None, :]
    z = Xc[..., 2]
    x = Xc[..., :2] / z.clamp(min=1e-6)[..., None]
    x = _distort(x, dist)
    fx, fy = K[..., 0, 0][..., None], K[..., 1, 1][..., None]
    cx, cy = K[..., 0, 2][..., None], K[..., 1, 2][..., None]
    uv = torch.stack([fx * x[..., 0] + cx, fy * x[..., 1] + cy], -1)
    return uv, z


def map_origin(x0: torch.Tensor, map_scale: float) -> torch.Tensor:
    """The ``origin`` that :func:`sample_views` needs for a crop's confidence maps.

    sleap-nn's confidence-map convention: index ``j`` of a crop whose top-left input
    pixel is ``x0`` is the input pixel ``x0 + j / map_scale`` (``map_scale =
    input_scale / output_stride``), with no half-pixel term. :func:`sample_views` maps
    a pixel ``u`` to ``(u - origin + 0.5) * map_scale - 0.5``, which is
    ``(u - x0) * map_scale`` for this origin.
    """
    return x0 - (0.5 / map_scale - 0.5)


def sample_views(
    maps: torch.Tensor,
    uv: torch.Tensor,
    z: torch.Tensor,
    origin: torch.Tensor,
    scale: torch.Tensor,
    hw: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Read each camera's maps bilinearly at the projected voxel centres.

    Args:
        maps: ``(B, V, C, h, w)`` maps (one crop per animal and camera).
        uv: ``(B, V, M, 2)`` projected voxel centres, in full-resolution pixels.
        z: ``(B, V, M)`` depths from :func:`project_points`.
        origin: ``(B, V, 2)`` see :func:`map_origin`.
        scale: ``(B, V)`` map pixels per full-resolution pixel.
        hw: The maps' ``(h, w)``.

    Returns:
        A tuple ``(vals, valid)``: ``(B, V, C, M)`` samples (0 outside the map or
        behind the camera) and ``(B, V, M)`` booleans (inside the map, in front).
    """
    B, V, C, h, w = maps.shape
    # full-res pixel centre u -> map pixel centre: (u - origin + 0.5) * scale - 0.5
    q = (uv - origin[:, :, None, :] + 0.5) * scale[:, :, None, None] - 0.5
    valid = (
        (z > 1e-3)
        & (q[..., 0] >= -0.5)
        & (q[..., 0] <= w - 0.5)
        & (q[..., 1] >= -0.5)
        & (q[..., 1] <= h - 0.5)
    )
    gx = (q[..., 0] + 0.5) / w * 2 - 1  # align_corners=False convention
    gy = (q[..., 1] + 0.5) / h * 2 - 1
    # Invalid samples are zeroed below anyway; moving them off the map first keeps a
    # non-finite projection (a point next to the image plane under strong k2/k3
    # distortion) from reaching grid_sample, where inf * 0 would give NaN. Valid
    # samples are untouched, so this matches the study's features exactly.
    gx = torch.where(valid, gx, torch.full_like(gx, -2.0))
    gy = torch.where(valid, gy, torch.full_like(gy, -2.0))
    g = torch.stack([gx, gy], -1).reshape(B * V, 1, -1, 2).to(maps.dtype)
    vals = F.grid_sample(
        maps.reshape(B * V, C, h, w),
        g,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    vals = vals.reshape(B, V, C, -1) * valid[:, :, None, :].to(vals.dtype)
    return vals, valid


def average_views(
    vals: torch.Tensor, valid: torch.Tensor, view_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Mean over the cameras that see each voxel (and are not dropped).

    Args:
        vals: ``(B, V, C, M)`` from :func:`sample_views`.
        valid: ``(B, V, M)`` from :func:`sample_views`.
        view_mask: ``(B, V)`` 1 for cameras in use, 0 for dropped or padded ones.

    Returns:
        ``(B, C, M)``; 0 where no camera sees the voxel.
    """
    w = valid.float()
    if view_mask is not None:
        w = w * view_mask[:, :, None]
    return (vals * w[:, :, None]).sum(1) / w.sum(1).clamp(min=1)[:, None]


# ---------------------------------------------------------------------------------------
# NumPy geometry (targets, matching, cube centres, scoring)
# ---------------------------------------------------------------------------------------


def np_project(
    X: np.ndarray, K: np.ndarray, dist: np.ndarray, R: np.ndarray, t: np.ndarray
) -> np.ndarray:
    """Project world points into one camera.

    Args:
        X: ``(..., 3)`` world points.
        K: ``(3, 3)`` intrinsics.
        dist: ``(5,)`` distortion ``[k1, k2, p1, p2, k3]`` (a scalar is ``k1`` alone).
        R: ``(3, 3)`` rotation.
        t: ``(3,)`` translation.

    Returns:
        ``(..., 2)`` distorted pixel coordinates.
    """
    k1, k2, p1, p2, k3 = _dist5(dist)
    Xc = X @ R.T + t
    x = Xc[..., :2] / Xc[..., 2:3]
    r2 = (x**2).sum(-1, keepdims=True)
    radial = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
    xx, yy = x[..., 0:1], x[..., 1:2]
    dx = 2 * p1 * xx * yy + p2 * (r2 + 2 * xx * xx)
    dy = p1 * (r2 + 2 * yy * yy) + 2 * p2 * xx * yy
    x = x * radial + np.concatenate([dx, dy], -1)
    return np.stack([K[0, 0] * x[..., 0] + K[0, 2], K[1, 1] * x[..., 1] + K[1, 2]], -1)


def np_undistort(
    uv: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    iters: int = 20,
    max_iters: int = 200,
    tol: float = 1e-12,
) -> np.ndarray:
    """Distorted pixels to undistorted normalised coordinates (fixed-point iteration).

    OpenCV's ``undistortPoints`` iteration. The first ``iters`` steps are the study's;
    with only ``k1`` set they are bitwise its ``x = xd / (1 + k1 |x|^2)``.

    Strong two-term lenses converge more slowly near the image corners. On calibrat3's
    ``fx, fy, cx, cy, k1, k2`` fits of the DoublePoke cameras (``k1`` about -0.40,
    ``k2`` about +0.22), 20 steps leave up to 0.05 px there. So a point whose next step
    still moves it by more than ``tol`` (normalised units) keeps iterating, up to
    ``max_iters`` steps in all. A point that had settled keeps its ``iters``-step value
    exactly, and so does one that never settles (outside the lens model's range).

    Args:
        uv: ``(..., 2)`` distorted pixels (NaN passes through).
        K: ``(3, 3)`` intrinsics.
        dist: ``(5,)`` distortion (or a scalar ``k1``).
        iters: Steps taken for every point (the study's 20).
        max_iters: Most steps for a point still moving after ``iters``.
        tol: A step smaller than this counts as settled.

    Returns:
        ``(..., 2)`` normalised coordinates.
    """
    k1, k2, p1, p2, k3 = _dist5(dist)
    xd = np.stack(
        [(uv[..., 0] - K[0, 2]) / K[0, 0], (uv[..., 1] - K[1, 2]) / K[1, 1]], -1
    )

    def step(x: np.ndarray, xd: np.ndarray) -> np.ndarray:
        r2 = (x**2).sum(-1, keepdims=True)
        radial = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
        xx, yy = x[..., 0:1], x[..., 1:2]
        dx = 2 * p1 * xx * yy + p2 * (r2 + 2 * xx * xx)
        dy = p1 * (r2 + 2 * yy * yy) + 2 * p2 * xx * yy
        return (xd - np.concatenate([dx, dy], -1)) / radial

    x = xd.copy()
    for _ in range(iters):
        x = step(x, xd)
    if max_iters <= iters:
        return x
    shape = x.shape
    x, xd = x.reshape(-1, 2), xd.reshape(-1, 2)
    settled = x.copy()
    idx = np.arange(len(x))
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        for _ in range(max_iters - iters):
            nxt = step(x[idx], xd[idx])
            moving = np.abs(nxt - x[idx]).max(-1) > tol  # NaN is not moving
            idx = idx[moving]
            if not len(idx):
                break
            x[idx] = nxt[moving]
    x[idx] = settled[idx]  # never settled: the study's value
    return x.reshape(shape)


def _dist5(dist) -> Tuple[float, float, float, float, float]:
    d = np.zeros(5, np.float64)
    dd = np.atleast_1d(np.asarray(dist, np.float64)).ravel()[:5]
    d[: len(dd)] = dd
    return tuple(float(v) for v in d)


def dlt(P: np.ndarray, xn: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted linear triangulation on undistorted normalised rays.

    Args:
        P: ``(V, 3, 4)`` normalised camera matrices ``[R | t]``.
        xn: ``(M, V, 2)`` normalised coordinates (NaN where unused).
        w: ``(M, V)`` weights (0 = unused view).

    Returns:
        ``(M, 3)`` points.
    """
    A1 = xn[..., 0:1] * P[None, :, 2, :] - P[None, :, 0, :]
    A2 = xn[..., 1:2] * P[None, :, 2, :] - P[None, :, 1, :]
    A = np.concatenate([A1, A2], 1) * np.concatenate([w, w], 1)[..., None]
    A = np.nan_to_num(A)
    _, _, Vt = np.linalg.svd(A)
    Xh = Vt[:, -1]
    return Xh[:, :3] / Xh[:, 3:4]


@attrs.define
class Triangulation:
    """Result of :func:`robust_triangulate`.

    Attributes:
        X: ``(M, 3)`` points (NaN with fewer than ``min_views`` views).
        used: ``(M, V)`` the views kept.
        err: ``(M,)`` mean reprojection error over the kept views (px).
        err_views: ``(M, V)`` per-view reprojection error (NaN for unused views).
    """

    X: np.ndarray
    used: np.ndarray
    err: np.ndarray
    err_views: np.ndarray

    @property
    def nview(self) -> np.ndarray:
        """``(M,)`` number of views kept."""
        return self.used.sum(-1)


def robust_triangulate(
    uv: np.ndarray,
    cams: CameraArrays,
    drop_px: float = DROP_PX,
    min_views: int = 2,
) -> Triangulation:
    """Robust DLT: triangulate, then drop the worst view while it is above ``drop_px``.

    The study's target rule (``slap_targets.py`` / ``rig_targets.py``): DLT on the
    undistorted rays of every view that sees the point; while the worst view's
    reprojection error exceeds ``drop_px`` and at least 3 views remain, that view is
    dropped and the point triangulated again.

    Args:
        uv: ``(M, V, 2)`` distorted pixels (NaN = not seen).
        cams: The ``V`` cameras.
        drop_px: Drop threshold (px).
        min_views: Views needed for a point.

    Returns:
        A :class:`Triangulation`.
    """
    uv = np.asarray(uv, np.float64)
    V = uv.shape[1]
    xn = np.stack(
        [np_undistort(uv[:, v], cams.K[v], cams.dist[v]) for v in range(V)], 1
    )
    return _robust_dlt(uv, xn, cams, drop_px, min_views)


def _robust_dlt(
    uv: np.ndarray,
    xn: np.ndarray,
    cams: CameraArrays,
    drop_px: float = DROP_PX,
    min_views: int = 2,
) -> Triangulation:
    """:func:`robust_triangulate` on rays already undistorted.

    ``xn`` is ``np_undistort`` of each view of ``uv`` (``(M, V, 2)`` both).
    Undistortion is point by point, so callers can undistort each camera's points
    once and triangulate many combinations of them. Views no point uses are not
    projected (their errors are NaN either way).
    """
    M, V, _ = uv.shape
    P = np.concatenate([cams.R, cams.t[:, :, None]], 2)
    w = np.isfinite(xn).all(-1).astype(np.float64)
    seen = np.nonzero(w.any(0))[0]
    X = np.full((M, 3), np.nan)
    err = np.full((M, V), np.nan)
    for _ in range(V):
        sel = w.sum(1) >= min_views
        if not sel.any():
            break
        Xs = np.full((M, 3), np.nan)
        Xs[sel] = dlt(P, xn[sel], w[sel])
        rep = np.full((M, V, 2), np.nan)
        for v in seen:
            rep[:, v] = np_project(Xs, cams.K[v], cams.dist[v], cams.R[v], cams.t[v])
        e = np.linalg.norm(rep - uv, axis=-1)
        e[w == 0] = np.nan
        X, err = Xs, e
        ee = np.where(np.isfinite(e), e, -1)
        worst = ee.argmax(1)
        we = ee[np.arange(M), worst]
        drop = (we > drop_px) & (w.sum(1) >= 3)
        if not drop.any():
            break
        w[np.nonzero(drop)[0], worst[drop]] = 0
    X[w.sum(1) < min_views] = np.nan
    used = w > 0
    with np.errstate(all="ignore"):
        errm = (
            _nanmean(np.where(used, err, np.nan), 1) if M else np.zeros(0, np.float64)
        )
    return Triangulation(
        X=X, used=used, err=errm, err_views=np.where(used, err, np.nan)
    )


def _nanmean(a: np.ndarray, axis: int) -> np.ndarray:
    """``np.nanmean`` without the empty-slice warning."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(a, axis)


def reliable_mask(
    X: np.ndarray,
    nview: np.ndarray,
    err: np.ndarray,
    min_views: int = RELIABLE_VIEWS,
    max_px: float = RELIABLE_PX,
) -> np.ndarray:
    """Points triangulated from at least ``min_views`` views below ``max_px`` error."""
    with np.errstate(invalid="ignore"):
        return np.isfinite(X).all(-1) & (nview >= min_views) & (err < max_px)


def cube_centers(
    X: np.ndarray, node_names: Sequence[str], center_nodes: Sequence[str]
) -> np.ndarray:
    """Cube centres from triangulated keypoints.

    The midpoint of ``center_nodes`` when all of them are finite; otherwise the mean
    of the finite keypoints when there are at least 3; otherwise NaN.

    Args:
        X: ``(N, J, 3)`` keypoints.
        node_names: The ``J`` node names.
        center_nodes: Nodes whose mean is the centre.

    Returns:
        ``(N, 3)`` centres.
    """
    X = np.asarray(X, np.float64)
    ci = [list(node_names).index(n) for n in center_nodes]
    cen = X[:, ci].mean(1)
    fin = np.isfinite(X).all(-1)
    with np.errstate(all="ignore"):
        m = _nanmean(np.where(fin[..., None], X, np.nan), 1)
    use = ~np.isfinite(cen).all(-1) & (fin.sum(-1) >= 3)
    cen[use] = m[use]
    return cen


# ---------------------------------------------------------------------------------------
# Matching detections across cameras (when a session has no instance groups)
# ---------------------------------------------------------------------------------------


def _median_distances(
    cams: CameraArrays, v: int, X: np.ndarray, inst: np.ndarray
) -> np.ndarray:
    """Median pixel distance between animals' reprojections and each instance.

    ``X`` is ``(A, J, 3)`` and ``inst`` ``(I, J, 2)``; returns ``(A, I)`` (NaN
    without a common node).
    """
    A, J = X.shape[:2]
    rp = np_project(
        X.reshape(-1, 3), cams.K[v], cams.dist[v], cams.R[v], cams.t[v]
    ).reshape(A, J, 2)
    d = np.linalg.norm(inst[None, :, :, :2] - rp[:, None], axis=-1)
    d = np.where(np.isfinite(X).all(-1)[:, None], d, np.nan)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(d, -1)


def _median_distance(
    cams: CameraArrays, v: int, X: np.ndarray, inst: np.ndarray
) -> np.ndarray:
    """:func:`_median_distances` for one animal: ``X`` ``(J, 3)``, returns ``(I,)``."""
    return _median_distances(cams, v, X[None], inst)[0]


def _gates(cams: CameraArrays, gate_px: float) -> np.ndarray:
    width = cams.size[:, 0].astype(np.float64)
    width = np.where(width > 0, width, 1280.0)
    return gate_px * width / 1280.0


def associate_frame(
    cams: CameraArrays,
    inst_by_view: Dict[int, np.ndarray],
    max_animals: int,
    pair_px: float = PAIR_PX,
    gate_px: float = GATE_PX,
    min_cams: int = MIN_CAMS,
) -> List[Dict[int, int]]:
    """Group one frame's detections into animals across cameras.

    The study's rule (``rig_targets.py``):
        1. Every camera pair proposes animals: instance ``i`` in camera ``a`` and
           ``j`` in camera ``b`` are triangulated on their common keypoints; a
           proposal needs a 2-view median reprojection error below ``pair_px``.
        2. Every other camera supports a proposal with its instance nearest to the
           proposal's reprojection (median over keypoints), when that is within the
           camera's gate (``gate_px`` x image width / 1280).
        3. Greedily, the proposal with the most supporting cameras (then the smallest
           summed distance) becomes an animal; its instances are taken; repeat. An
           animal needs ``min_cams`` cameras; at most ``max_animals`` are kept.
        4. Each animal is triangulated, and every camera is matched again to the 3D
           animals (Hungarian on the same median distance, same gate).

    Args:
        cams: The session's cameras.
        inst_by_view: ``{view: (I, J, 2) instances}`` in a shared node order (NaN =
            missing node). Instances with fewer than 3 finite nodes are ignored.
        max_animals: Most animals to return.
        pair_px: Step 1 threshold.
        gate_px: Step 2 and 4 gate at a 1280 px wide image.
        min_cams: Cameras an animal needs.

    Returns:
        One ``{view: instance index}`` per animal (indices into ``inst_by_view``).
    """
    V = cams.n_views
    gates = _gates(cams, gate_px)
    ibc = {}
    for v, x in inst_by_view.items():
        if x is None or len(x) == 0:
            continue
        keep = np.nonzero(np.isfinite(x[:, :, :2]).all(-1).sum(-1) >= 3)[0]
        if len(keep):
            ibc[v] = (x[keep, :, :2].astype(np.float64), keep)
    if len(ibc) < min_cams or max_animals <= 0:
        return []
    J = next(iter(ibc.values()))[0].shape[1]
    # Each camera's detections are undistorted once (undistortion is point by point),
    # every camera pair's candidates are triangulated in one batch, and each camera
    # scores every proposal in one batch. The order of proposals and of the summed
    # distances is the per-candidate loop's, so ties break the same way.
    xn_by = {v: np_undistort(x, cams.K[v], cams.dist[v]) for v, (x, _) in ibc.items()}
    cand = [
        (a, b, i, j)
        for a, b in combinations(sorted(ibc), 2)
        for i in range(len(ibc[a][0]))
        for j in range(len(ibc[b][0]))
    ]
    uv = np.full((len(cand), J, V, 2), np.nan)
    xn = np.full((len(cand), J, V, 2), np.nan)
    for k, (a, b, i, j) in enumerate(cand):
        uv[k, :, a], uv[k, :, b] = ibc[a][0][i], ibc[b][0][j]
        xn[k, :, a], xn[k, :, b] = xn_by[a][i], xn_by[b][j]
    tr = _robust_dlt(uv.reshape(-1, V, 2), xn.reshape(-1, V, 2), cams)
    Xc = tr.X.reshape(len(cand), J, 3)
    ev = tr.err_views.reshape(len(cand), J, V)
    accepted = []
    for k, (a, b, i, j) in enumerate(cand):
        if np.isfinite(Xc[k]).all(-1).sum() < 3:
            continue
        e2 = _nanmedian(_nanmean(ev[k][:, [a, b]], 1))
        if e2 < pair_px:
            accepted.append(k)
    dist_by = (
        {c: _median_distances(cams, c, Xc[accepted], ibc[c][0]) for c in ibc}
        if accepted
        else {}
    )
    props = []
    for p, k in enumerate(accepted):
        a, b, i, j = cand[k]
        sup, tot = {a: i, b: j}, 0.0
        for c in ibc:
            if c in (a, b):
                continue
            d = dist_by[c][p]
            if np.isfinite(d).any() and np.nanmin(d) < gates[c]:
                sup[c] = int(np.nanargmin(d))
                tot += float(np.nanmin(d))
        props.append((len(sup), -tot, sup))
    animals: List[Dict[int, int]] = []
    taken = {v: set() for v in range(V)}
    for n, _, sup in sorted(props, key=lambda p: (p[0], p[1]), reverse=True):
        if len(animals) >= max_animals:
            break
        if n < min_cams or any(i in taken[v] for v, i in sup.items()):
            continue
        animals.append(sup)
        for v, i in sup.items():
            taken[v].add(i)
    if not animals:
        return []
    # step 4: triangulate each animal, then match every camera again to the 3D animals
    Xa = []
    for sup in animals:
        uv = np.full((J, V, 2), np.nan)
        xn = np.full((J, V, 2), np.nan)
        for v, i in sup.items():
            uv[:, v], xn[:, v] = ibc[v][0][i], xn_by[v][i]
        Xa.append(_robust_dlt(uv, xn, cams).X)
    out: List[Dict[int, int]] = [dict() for _ in animals]
    for v, (inst, keep) in ibc.items():
        d = np.stack([_median_distance(cams, v, X, inst) for X in Xa])
        d = np.nan_to_num(d, nan=1e9)
        r, c = linear_sum_assignment(d)
        for ai, ii in zip(r, c):
            if d[ai, ii] < gates[v]:
                out[ai][v] = int(keep[ii])
    return out


def _nanmedian(a) -> float:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return float(np.nanmedian(a))


# ---------------------------------------------------------------------------------------
# Floor normal
# ---------------------------------------------------------------------------------------


@attrs.define
class FloorFit:
    """A fitted floor normal.

    Attributes:
        up: ``(3,)`` unit normal, pointing to the cameras' side.
        method: How it was found (``"given"``, ``"metadata"``, ``"up_fit_nodes"``,
            ``"lowest_nodes"``).
        inlier_fraction: Share of the points within the band (fits only).
        tilt_degrees: Angle to the rough fit it started from (``lowest_nodes`` only).
        n_points: Points the fit used.
    """

    up: np.ndarray
    method: str
    inlier_fraction: Optional[float] = None
    tilt_degrees: Optional[float] = None
    n_points: int = 0


def _toward(normal: np.ndarray, points: np.ndarray, cam_centers: np.ndarray):
    n = normal / np.linalg.norm(normal)
    if np.mean((cam_centers - points.mean(0)) @ n) < 0:
        n = -n
    return n


def _plane_lsq(points: np.ndarray) -> np.ndarray:
    p = points - points.mean(0)
    _, _, vt = np.linalg.svd(p, full_matrices=False)
    return vt[-1]


def fit_floor_normal(
    points: np.ndarray,
    cam_centers: np.ndarray,
    band: float = 4.0,
    iters: int = 500,
    seed: int = 0,
) -> Tuple[np.ndarray, float]:
    """RANSAC plane through points on the floor, refit by least squares on inliers.

    Args:
        points: ``(N, 3)`` points that lie on (or near) the floor.
        cam_centers: ``(V, 3)`` camera centres: the normal points to their side.
        band: Inlier distance from the plane (world units).
        iters: RANSAC samples.
        seed: Random seed.

    Returns:
        A tuple ``(normal, inlier_fraction)``.
    """
    p = np.asarray(points, np.float64)
    p = p[np.isfinite(p).all(-1)]
    if len(p) < 3:
        raise ValueError(f"A floor plane needs at least 3 points; got {len(p)}.")
    rng = np.random.default_rng(seed)
    best_n, best_in = _plane_lsq(p), -1
    for _ in range(iters):
        s = p[rng.choice(len(p), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            continue
        n = n / nn
        inl = int((np.abs((p - s[0]) @ n) < band).sum())
        if inl > best_in:
            best_in, best_n, best_p = inl, n, s[0]
    if best_in < 0:  # degenerate sample set: least squares on everything
        best_p = p.mean(0)
    inliers = np.abs((p - best_p) @ best_n) < band
    if inliers.sum() >= 3:
        best_n = _plane_lsq(p[inliers])
    return _toward(best_n, p, cam_centers), float(inliers.mean())


def _up_from_metadata(session: sio.RecordingSession) -> Optional[np.ndarray]:
    for md in (session.metadata, getattr(session.camera_group, "metadata", {}) or {}):
        for key in ("up", "floor_normal"):
            v = md.get(key) if isinstance(md, dict) else None
            if v is not None:
                v = np.asarray(v, np.float64).ravel()
                if v.shape == (3,) and np.isfinite(v).all() and np.linalg.norm(v) > 0:
                    return v / np.linalg.norm(v)
    return None


def resolve_up(
    session: sio.RecordingSession,
    cams: CameraArrays,
    X: np.ndarray,
    reliable: np.ndarray,
    node_names: Sequence[str],
    up: Optional[Sequence[float]] = None,
    up_fit_nodes: Optional[Sequence[str]] = None,
    band: float = 4.0,
) -> FloorFit:
    """The floor normal of a session, in order of preference.

    1. ``up`` given (the config's), else the session's metadata (``metadata["up"]`` or
       ``metadata["floor_normal"]`` on the session or its camera group).
    2. A RANSAC plane through the reliable points of ``up_fit_nodes`` (nodes that touch
       the floor: tail tips, paws).
    3. A RANSAC plane through each animal-frame's lowest reliable node. "Lowest" is
       measured along a rough normal (the smallest-variance axis of all reliable
       points); the plane is fitted, the lowest nodes picked again along it, and the
       plane fitted once more. It warns when fewer than 60 % of the points are inliers
       or the normal is more than 3 degrees from the rough one (rearing animals, a
       ramp, an uneven floor). Body centres are never used: rearing tilts a fit to them.

    The normal always points to the cameras' side.

    Args:
        session: The recording session (for its metadata).
        cams: Its cameras.
        X: ``(N, J, 3)`` triangulated keypoints of its animal-frames.
        reliable: ``(N, J)`` reliable points (see :func:`reliable_mask`).
        node_names: The ``J`` node names.
        up: An explicit normal.
        up_fit_nodes: Floor-touching nodes.
        band: RANSAC inlier band (world units).

    Returns:
        A :class:`FloorFit`.
    """
    cc = cams.centers
    if up is not None:
        n = np.asarray(up, np.float64).ravel()
        return FloorFit(up=n / np.linalg.norm(n), method="given")
    md = _up_from_metadata(session)
    if md is not None:
        return FloorFit(up=md, method="metadata")
    X = np.asarray(X, np.float64)
    rel = np.asarray(reliable, bool) & np.isfinite(X).all(-1)
    if up_fit_nodes:
        idx = [list(node_names).index(n) for n in up_fit_nodes]
        pts = X[:, idx][rel[:, idx]]
        n, frac = fit_floor_normal(pts, cc, band)
        _warn_floor(frac, None, len(pts), "up_fit_nodes")
        return FloorFit(
            up=n, method="up_fit_nodes", inlier_fraction=frac, n_points=len(pts)
        )
    allp = X[rel]
    if len(allp) < 3:
        raise ValueError(
            "Cannot fit a floor normal: fewer than 3 reliable 3D points. Set "
            "data_config.multiview.up."
        )
    rough = _toward(_plane_lsq(allp), allp, cc)
    n = rough
    has = rel.any(-1)
    frac, pts = 0.0, allp
    for _ in range(2):
        h = np.where(rel, X @ n, np.inf)
        low = h.argmin(-1)
        pts = X[np.arange(len(X)), low][has]
        n, frac = fit_floor_normal(pts, cc, band)
    tilt = float(np.degrees(np.arccos(np.clip(abs(float(n @ rough)), -1.0, 1.0))))
    _warn_floor(frac, tilt, len(pts), "lowest_nodes")
    return FloorFit(
        up=n,
        method="lowest_nodes",
        inlier_fraction=frac,
        tilt_degrees=tilt,
        n_points=len(pts),
    )


def _warn_floor(frac: float, tilt: Optional[float], n: int, how: str):
    msg = f"Floor normal from {how}: {n} points, {frac:.0%} inliers"
    if tilt is not None:
        msg += f", {tilt:.1f} degrees from the rough fit"
    if frac < 0.6 or (tilt is not None and tilt > 3.0):
        logger.warning(
            msg + ". This floor fit looks poor (a ramp, an uneven floor or much "
            "rearing?). Set data_config.multiview.up or up_fit_nodes."
        )
    else:
        logger.info(msg)


# ---------------------------------------------------------------------------------------
# A session's instance groups as training / scoring targets
# ---------------------------------------------------------------------------------------


@attrs.define
class SessionData:
    """A session's animal-frames: 2D per camera, 3D targets and their reliability.

    One row per instance group on a frame group.

    Attributes:
        session: The recording session.
        cameras: Its cameras.
        videos: Each camera's video (``None`` where the session has none).
        node_names: Node order of every array.
        frame_idx: ``(N,)`` frame index.
        group_index: ``(N,)`` index of the instance group in its frame group.
        pts2d: ``(N, V, J, 2)`` 2D points per camera (NaN = missing).
        X: ``(N, J, 3)`` 3D targets (NaN = none).
        nview: ``(N, J)`` views of the robust DLT.
        err: ``(N, J)`` its mean reprojection error (px).
        given_3d: ``(N,)`` the target is the group's ``instance_3d``, not the DLT.
        dlt: ``(N, J, 3)`` the robust DLT of ``pts2d`` (equal to ``X`` where
            ``given_3d`` is False).
    """

    session: sio.RecordingSession
    cameras: CameraArrays
    videos: List[Optional[sio.Video]]
    node_names: List[str]
    frame_idx: np.ndarray
    group_index: np.ndarray
    pts2d: np.ndarray
    X: np.ndarray
    nview: np.ndarray
    err: np.ndarray
    given_3d: np.ndarray
    dlt: np.ndarray

    def __len__(self) -> int:
        """Number of animal-frames."""
        return len(self.frame_idx)

    def target_mask(self, min_views: int = 2, max_reproj_px: float = 10.0):
        """``(N, J)`` points to train on.

        A given ``instance_3d`` point counts when finite; a triangulated one needs
        ``min_views`` views and a mean reprojection error below ``max_reproj_px``.
        """
        fin = np.isfinite(self.X).all(-1)
        with np.errstate(invalid="ignore"):
            tri = fin & (self.nview >= min_views) & (self.err < max_reproj_px)
        return np.where(self.given_3d[:, None], fin, tri)

    def reliable(self) -> np.ndarray:
        """``(N, J)`` points to score on (given 3D, or >= 3 views below 5 px)."""
        fin = np.isfinite(self.X).all(-1)
        return np.where(
            self.given_3d[:, None],
            fin,
            reliable_mask(self.X, self.nview, self.err),
        )


def _instance_points(inst, node_names: Sequence[str]) -> np.ndarray:
    """``(J, 2)`` points of an instance in ``node_names`` order (NaN = missing)."""
    pts = np.asarray(inst.numpy(), np.float64)[:, :2]
    names = inst.skeleton.node_names
    out = np.full((len(node_names), 2), np.nan)
    for j, n in enumerate(node_names):
        if n in names:
            out[j] = pts[names.index(n)]
    return out


def _prefer_user(lf: Optional[sio.LabeledFrame], inst):
    """A user instance made from ``inst`` when the frame has one, else ``inst``."""
    if lf is None or not isinstance(inst, sio.PredictedInstance):
        return inst
    for u in lf.user_instances:
        if getattr(u, "from_predicted", None) is inst:
            return u
    return inst


def extract_session_data(
    session: sio.RecordingSession,
    node_names: Sequence[str],
    target_source: str = "auto",
    drop_px: float = DROP_PX,
) -> SessionData:
    """A session's instance groups as 3D targets.

    Each instance group of each frame group is one animal-frame. Its 2D points are
    its instance in each camera (a user instance made from a predicted one wins). Its
    3D target is its ``instance_3d`` when set and ``target_source`` is ``"auto"`` or
    ``"instance_3d"``; otherwise the robust DLT of its 2D points (``"triangulate"``
    always triangulates).

    Args:
        session: A recording session with frame groups.
        node_names: The node order to use (the volume model's ``part_names``).
        target_source: ``"auto"``, ``"instance_3d"`` or ``"triangulate"``.
        drop_px: The robust DLT's drop threshold.

    Returns:
        A :class:`SessionData` (empty when the session has no instance groups).
    """
    if target_source not in ("auto", "instance_3d", "triangulate"):
        raise ValueError(
            f"target_source must be auto, instance_3d or triangulate; got "
            f"{target_source!r}."
        )
    cams = CameraArrays.from_session(session)
    cam_objs = list(session.camera_group.cameras)
    V, J = len(cam_objs), len(node_names)
    videos = [session.get_video(c) for c in cam_objs]
    rows_f, rows_g, pts, x3, given = [], [], [], [], []
    for fidx in sorted(session.frame_groups):
        fg = session.frame_groups[fidx]
        for gi, ig in enumerate(fg.instance_groups):
            p = np.full((V, J, 2), np.nan)
            for v, cam in enumerate(cam_objs):
                inst = ig.get_instance(cam)
                if inst is None:
                    continue
                inst = _prefer_user(fg.get_frame(cam), inst)
                p[v] = _instance_points(inst, node_names)
            t3 = np.full((J, 3), np.nan)
            has3 = False
            i3 = ig.instance_3d
            if target_source != "triangulate" and i3 is not None and not i3.is_empty:
                names = i3.skeleton.node_names
                pp = i3.numpy()
                for j, n in enumerate(node_names):
                    if n in names:
                        t3[j] = pp[names.index(n)]
                has3 = bool(np.isfinite(t3).all(-1).any())
            if target_source == "instance_3d" and not has3:
                continue
            rows_f.append(int(fidx))
            rows_g.append(gi)
            pts.append(p)
            x3.append(t3)
            given.append(has3)
    N = len(rows_f)
    pts2d = np.stack(pts) if N else np.zeros((0, V, J, 2))
    if N:
        uv = np.moveaxis(pts2d, 1, 2).reshape(-1, V, 2)
        tr = robust_triangulate(uv, cams, drop_px=drop_px)
        dlt_X = tr.X.reshape(N, J, 3)
        nview = tr.nview.reshape(N, J).astype(np.int16)
        err = tr.err.reshape(N, J)
    else:
        dlt_X = np.zeros((0, J, 3))
        nview = np.zeros((0, J), np.int16)
        err = np.zeros((0, J))
    given_3d = np.asarray(given, bool)
    X = np.where(given_3d[:, None, None], np.stack(x3) if N else dlt_X, dlt_X)
    return SessionData(
        session=session,
        cameras=cams,
        videos=videos,
        node_names=list(node_names),
        frame_idx=np.asarray(rows_f, np.int64),
        group_index=np.asarray(rows_g, np.int64),
        pts2d=pts2d.astype(np.float32),
        X=X.astype(np.float64),
        nview=nview,
        err=err.astype(np.float64),
        given_3d=given_3d,
        dlt=dlt_X.astype(np.float64),
    )
