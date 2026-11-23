"""Tests for the volumetric model's multi-camera geometry (`sleap_nn.data.multiview`)."""

import math

import cv2
import numpy as np
import pytest
import sleap_io as sio
import torch
import torch.nn.functional as F

from sleap_nn.data import multiview as MV
from tests.fixtures.multiview import (
    NODES,
    make_cameras,
    make_session_labels,
    project as cv_project,
    random_poses,
)

# ---------------------------------------------------------------------------------------
# The 2026-11-11 study's own formulas (vol3d.py / slap_targets.py), k1-only, kept
# verbatim as references: with only k1 set, the port must reproduce them bit for bit.
# ---------------------------------------------------------------------------------------


def _study_project_torch(Xw, K, k1, R, t):
    Xc = torch.einsum("bvij,bmj->bvmi", R, Xw) + t[:, :, None, :]
    z = Xc[..., 2]
    x = Xc[..., :2] / z.clamp(min=1e-6)[..., None]
    r2 = (x * x).sum(-1, keepdim=True)
    x = x * (1 + k1[:, :, None, None] * r2)
    fx, fy = K[..., 0, 0][..., None], K[..., 1, 1][..., None]
    cx, cy = K[..., 0, 2][..., None], K[..., 1, 2][..., None]
    uv = torch.stack([fx * x[..., 0] + cx, fy * x[..., 1] + cy], -1)
    return uv, z


def _study_np_project(X, K, k1, R, t):
    Xc = X @ R.T + t
    x = Xc[..., :2] / Xc[..., 2:3]
    r2 = (x**2).sum(-1, keepdims=True)
    x = x * (1 + k1 * r2)
    return np.stack([K[0, 0] * x[..., 0] + K[0, 2], K[1, 1] * x[..., 1] + K[1, 2]], -1)


def _study_undistort(uv, K, k1, iters=20):
    xd = np.stack(
        [(uv[..., 0] - K[0, 2]) / K[0, 0], (uv[..., 1] - K[1, 2]) / K[1, 1]], -1
    )
    x = xd.copy()
    for _ in range(iters):
        r2 = (x**2).sum(-1, keepdims=True)
        x = xd / (1 + k1 * r2)
    return x


def _study_dlt(P, xn, w):
    A1 = xn[..., 0:1] * P[None, :, 2, :] - P[None, :, 0, :]
    A2 = xn[..., 1:2] * P[None, :, 2, :] - P[None, :, 1, :]
    A = np.concatenate([A1, A2], 1) * np.concatenate([w, w], 1)[..., None]
    A = np.nan_to_num(A)
    _, _, Vt = np.linalg.svd(A)
    Xh = Vt[:, -1]
    return Xh[:, :3] / Xh[:, 3:4]


def _study_targets(uvf, K, k1, R, t, drop_px=12.0):
    """slap_targets.py's robust DLT loop (the SLAP targets)."""
    V = len(K)
    Pn = np.concatenate([R, t[:, :, None]], 2)
    xn = np.stack([_study_undistort(uvf[:, v], K[v], k1[v]) for v in range(V)], 1)
    w = np.isfinite(xn).all(-1).astype(np.float64)
    X = np.full((len(uvf), 3), np.nan)
    err = np.full((len(uvf), V), np.nan)
    for _ in range(V):
        sel = w.sum(1) >= 2
        Xs = np.full((len(uvf), 3), np.nan)
        Xs[sel] = _study_dlt(Pn, xn[sel], w[sel])
        rep = np.stack(
            [_study_np_project(Xs, K[v], k1[v], R[v], t[v]) for v in range(V)], 1
        )
        e = np.linalg.norm(rep - uvf, axis=-1)
        e[w == 0] = np.nan
        X, err = Xs, e
        worst = np.nanargmax(np.where(np.isfinite(e), e, -1), 1)
        we = e[np.arange(len(e)), worst]
        with np.errstate(invalid="ignore"):
            drop = (we > drop_px) & (w.sum(1) >= 3)
        if not drop.any():
            break
        w[np.nonzero(drop)[0], worst[drop]] = 0
    nview = w.sum(1)
    X[nview < 2] = np.nan
    used = w > 0
    with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore")
        errm = np.nanmean(np.where(used, err, np.nan), 1)
    return X, nview, errm


def _cams(n=4, distortion=True, seed=0):
    return MV.CameraArrays.from_cameras(
        make_cameras(n, distortion=distortion, seed=seed)
    )


def _k1_only(cams):
    d = cams.dist.copy()
    d[:, 1:] = 0
    return MV.CameraArrays(cams.names, cams.K, d, cams.R, cams.t, cams.size)


# ---------------------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------------------


def test_camera_arrays_from_session():
    labels, _ = make_session_labels(n_frames=1)
    cams = MV.CameraArrays.from_session(labels.sessions[0])
    assert cams.n_views == 4 and cams.names == ["cam0", "cam1", "cam2", "cam3"]
    assert cams.K.shape == (4, 3, 3) and cams.dist.shape == (4, 5)
    assert np.allclose(cams.R @ np.swapaxes(cams.R, 1, 2), np.eye(3), atol=1e-12)
    # every camera centre is ~700 mm out and above the floor, looking at the origin
    assert np.all(np.linalg.norm(cams.centers[:, :2], axis=1) > 600)
    assert np.all(cams.centers[:, 2] > 400)
    assert cams.size.tolist() == [[640, 480]] * 4
    sub = cams.subset([2, 0])
    assert sub.names == ["cam2", "cam0"] and np.array_equal(sub.K[1], cams.K[0])


def test_projection_matches_opencv():
    """NumPy and torch projection equal cv2.projectPoints with all 5 distortion terms."""
    sio_cams = make_cameras(4, distortion=True, seed=3)
    cams = MV.CameraArrays.from_cameras(sio_cams)
    X = random_poses(50, seed=1).reshape(-1, 3)
    T = cams.to_torch(dtype=torch.float64)
    uv_t, z = MV.project_points(
        torch.as_tensor(X)[None],
        T["K"][None],
        T["dist"][None],
        T["R"][None],
        T["t"][None],
    )
    for v, cam in enumerate(sio_cams):
        ref = cv_project(cam, X)
        mine = MV.np_project(X, cams.K[v], cams.dist[v], cams.R[v], cams.t[v])
        assert np.abs(mine - ref).max() < 1e-9
        assert np.abs(uv_t[0, v].numpy() - ref).max() < 1e-9
    assert (z > 0).all()


def test_undistort_inverts_projection_and_matches_opencv():
    sio_cams = make_cameras(3, distortion=True, seed=5)
    cams = MV.CameraArrays.from_cameras(sio_cams)
    X = random_poses(20, seed=2).reshape(-1, 3)
    for v in range(3):
        uv = MV.np_project(X, cams.K[v], cams.dist[v], cams.R[v], cams.t[v])
        xn = MV.np_undistort(uv, cams.K[v], cams.dist[v])
        Xc = X @ cams.R[v].T + cams.t[v]
        assert np.abs(xn - Xc[:, :2] / Xc[:, 2:]).max() < 1e-9
        ref = cv2.undistortPoints(
            uv.reshape(-1, 1, 2), cams.K[v], cams.dist[v]
        ).reshape(-1, 2)
        assert np.abs(xn - ref).max() < 1e-5


def test_k1_only_is_bitwise_the_study():
    """With only k1 set, projection and undistortion are the study's arithmetic."""
    cams = _k1_only(_cams(4, seed=7))
    X = random_poses(30, seed=4).reshape(-1, 3)
    for v in range(4):
        a = MV.np_project(X, cams.K[v], cams.dist[v], cams.R[v], cams.t[v])
        b = _study_np_project(X, cams.K[v], cams.dist[v, 0], cams.R[v], cams.t[v])
        assert np.array_equal(a, b)
        c = MV.np_undistort(a, cams.K[v], cams.dist[v])
        d = _study_undistort(a, cams.K[v], cams.dist[v, 0])
        assert np.array_equal(c, d)
    T = cams.to_torch()
    Xw = torch.as_tensor(X, dtype=torch.float32)[None].expand(2, -1, -1)
    args = [T[k][None].expand(2, *T[k].shape) for k in ("K", "dist", "R", "t")]
    uv, z = MV.project_points(Xw, *args)
    uv_s, z_s = _study_project_torch(Xw, args[0], args[1][..., 0], args[2], args[3])
    assert torch.equal(uv, uv_s) and torch.equal(z, z_s)


# ---------------------------------------------------------------------------------------
# Cube and unprojection
# ---------------------------------------------------------------------------------------


def test_cube_offsets_and_basis():
    off = MV.cube_offsets(4, 8.0)
    assert off.shape == (64, 3)
    assert torch.allclose(off[0], torch.tensor([-3.0, -3.0, -3.0]))
    assert torch.allclose(off[1], torch.tensor([-3.0, -3.0, -1.0]))  # C order
    up = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.6, 0.8]])
    yaw = torch.tensor([0.0, 1.0])
    B = MV.cube_basis(up, yaw)
    for b in range(2):
        assert torch.allclose(B[b] @ B[b].T, torch.eye(3), atol=1e-6)
        assert torch.allclose(B[b, 2], up[b])
        assert torch.det(B[b]) > 0
    # the yaw turns e1 about up
    B0 = MV.cube_basis(up[1:], torch.zeros(1))
    assert torch.allclose(
        B[1, 0], math.cos(1.0) * B0[0, 0] + math.sin(1.0) * B0[0, 1], atol=1e-6
    )
    grid = MV.cube_world(torch.tensor([[1.0, 2.0, 3.0]]), B0, off)
    assert torch.allclose(grid.mean(1), torch.tensor([[1.0, 2.0, 3.0]]), atol=1e-5)


def _bilinear_zero_pad(m, qx, qy):
    """Explicit bilinear read of map m (h, w) at map-pixel coords (centres at ints)."""
    h, w = m.shape
    x0, y0 = math.floor(qx), math.floor(qy)
    out = 0.0
    for dy in (0, 1):
        for dx in (0, 1):
            xi, yi = x0 + dx, y0 + dy
            wt = (1 - abs(qx - xi)) * (1 - abs(qy - yi))
            if 0 <= xi < w and 0 <= yi < h:
                out += wt * m[yi, xi]
    return out


@pytest.mark.parametrize("seed", [0, 1])
def test_unprojection_matches_numpy_reference(seed):
    """project -> sample -> average equals a per-voxel NumPy loop (cv2 projection)."""
    rng = np.random.default_rng(seed)
    sio_cams = make_cameras(3, distortion=True, seed=seed + 10)
    cams = MV.CameraArrays.from_cameras(sio_cams)
    B, V, C, h, w, n, size = 2, 3, 2, 24, 20, 5, 160.0
    maps = rng.random((B, V, C, h, w)).astype(np.float32)
    centers = rng.uniform(-60, 60, size=(B, 3))
    centers[:, 2] = 20
    yaw = rng.uniform(0, 2 * np.pi, size=B)
    up = np.array([[0.0, 0.0, 1.0], [0.05, -0.03, 1.0]])
    up /= np.linalg.norm(up, axis=1, keepdims=True)
    # each camera's crop: top-left x0 near the projected centre, 1/4 map scale
    map_scale = 0.25
    x0 = np.zeros((B, V, 2))
    for b in range(B):
        for v in range(V):
            c = cv_project(sio_cams[v], centers[b])
            x0[b, v] = np.round(c) - np.array([w, h]) / map_scale / 2
    vmask = np.array([[1, 1, 1], [1, 0, 1]], np.float32)

    T = cams.to_torch(dtype=torch.float32)
    basis = MV.cube_basis(
        torch.as_tensor(up, dtype=torch.float32),
        torch.as_tensor(yaw, dtype=torch.float32),
    )
    grid = MV.cube_world(
        torch.as_tensor(centers, dtype=torch.float32), basis, MV.cube_offsets(n, size)
    )
    uv, z = MV.project_points(
        grid, *[T[k][None].expand(B, *T[k].shape) for k in ("K", "dist", "R", "t")]
    )
    origin = MV.map_origin(torch.as_tensor(x0, dtype=torch.float32), map_scale)
    vals, valid = MV.sample_views(
        torch.as_tensor(maps), uv, z, origin, torch.full((B, V), map_scale), (h, w)
    )
    agg = MV.average_views(vals, valid, torch.as_tensor(vmask)).numpy()

    G = grid.numpy().astype(np.float64)
    ref = np.zeros((B, C, n**3))
    for b in range(B):
        for m in range(n**3):
            num, den = np.zeros(C), 0.0
            for v in range(V):
                if not vmask[b, v]:
                    continue
                Xc = cams.R[v] @ G[b, m] + cams.t[v]
                if Xc[2] <= 1e-3:
                    continue
                u = cv_project(sio_cams[v], G[b, m])
                qx, qy = (u - x0[b, v]) * map_scale
                if not (-0.5 <= qx <= w - 0.5 and -0.5 <= qy <= h - 0.5):
                    continue
                num += [_bilinear_zero_pad(maps[b, v, c], qx, qy) for c in range(C)]
                den += 1
            ref[b, :, m] = num / max(den, 1)
    assert np.abs(agg - ref).max() < 1e-5
    # the test is not vacuous: most voxels are seen
    assert (valid.float().mean() > 0.5).item()


def test_gaussian_maps_unproject_to_the_point():
    """Gaussians at each camera's projection of a point peak within a voxel of it."""
    sio_cams = make_cameras(4, distortion=True, seed=2)
    cams = MV.CameraArrays.from_cameras(sio_cams)
    P = np.array([12.0, -7.0, 18.0])
    n, size, map_scale = 32, 128.0, 0.5
    h = w = 96
    center = P + np.array([9.0, -5.0, 3.0])  # the cube is not centred on the point
    maps = np.zeros((1, 4, 1, h, w), np.float32)
    x0 = np.zeros((1, 4, 2))
    yy, xx = np.mgrid[0:h, 0:w]
    for v, cam in enumerate(sio_cams):
        x0[0, v] = np.round(cv_project(cam, center)) - np.array([w, h]) / map_scale / 2
        q = (cv_project(cam, P) - x0[0, v]) * map_scale
        maps[0, v, 0] = np.exp(-((xx - q[0]) ** 2 + (yy - q[1]) ** 2) / (2 * 1.5**2))
    T = cams.to_torch()
    basis = MV.cube_basis(torch.tensor([[0.0, 0.0, 1.0]]), torch.tensor([0.4]))
    grid = MV.cube_world(
        torch.as_tensor(center, dtype=torch.float32)[None],
        basis,
        MV.cube_offsets(n, size),
    )
    uv, z = MV.project_points(grid, *[T[k][None] for k in ("K", "dist", "R", "t")])
    vals, valid = MV.sample_views(
        torch.as_tensor(maps),
        uv,
        z,
        MV.map_origin(torch.as_tensor(x0, dtype=torch.float32), map_scale),
        torch.full((1, 4), map_scale),
        (h, w),
    )
    agg = MV.average_views(vals, valid)[0, 0]
    best = grid[0, int(agg.argmax())].numpy()
    assert np.linalg.norm(best - P) <= size / n * math.sqrt(3)


def test_sample_views_is_finite_next_to_the_image_plane():
    """A voxel at the camera's image plane under strong k3 must not poison the mean."""
    cams = _cams(2, seed=1)
    cams.dist[:, 4] = 5.0  # strong k3: r^6 overflows float32 for points near z = 0
    T = cams.to_torch()
    # one point exactly in camera 0's image plane, one ordinary point
    c0 = cams.centers[0]
    R0 = cams.R[0]
    X = np.stack([c0 + 50 * R0[0], np.zeros(3)])
    Xw = torch.as_tensor(X, dtype=torch.float32)[None]
    uv, z = MV.project_points(Xw, *[T[k][None] for k in ("K", "dist", "R", "t")])
    maps = torch.ones(1, 2, 1, 8, 8)
    vals, valid = MV.sample_views(
        maps, uv, z, torch.zeros(1, 2, 2), torch.full((1, 2), 1.0), (8, 8)
    )
    agg = MV.average_views(vals, valid)
    assert torch.isfinite(agg).all()


# ---------------------------------------------------------------------------------------
# Robust DLT, cube centres, matching across cameras
# ---------------------------------------------------------------------------------------


def test_robust_triangulate_recovers_points_and_drops_outliers():
    sio_cams = make_cameras(5, distortion=True, seed=4)
    cams = MV.CameraArrays.from_cameras(sio_cams)
    X = random_poses(10, seed=3).reshape(-1, 3)
    uv = np.stack([cv_project(c, X) for c in sio_cams], 1)  # (M, V, 2)
    # an outlier view. (The study's DLT is unnormalised, so a far larger outlier can
    # pull the first solution enough that a good view is dropped instead: with 5
    # cameras an 80 px outlier is removed 85 % of the time. The port keeps that
    # method exactly, so its targets match the study's.)
    uv[0, 2] += 30.0
    uv[1, 1:] = np.nan  # seen once: no point
    uv[2, 2:] = np.nan  # seen twice: kept
    tr = MV.robust_triangulate(uv, cams)
    assert np.isnan(tr.X[1]).all() and tr.nview[1] == 1
    ok = np.ones(len(X), bool)
    ok[1] = False
    assert np.abs(tr.X[ok] - X[ok]).max() < 1e-6
    assert not tr.used[0, 2] and tr.nview[0] == 4 and tr.err[0] < 1e-6
    assert tr.nview[2] == 2


def test_robust_triangulate_is_bitwise_the_study_targets():
    """k1-only: X, nview and err equal slap_targets.py's loop exactly."""
    sio_cams = make_cameras(6, distortion=True, seed=8)
    cams = _k1_only(MV.CameraArrays.from_cameras(sio_cams))
    rng = np.random.default_rng(0)
    X = random_poses(40, seed=9).reshape(-1, 3)
    uv = np.stack(
        [
            MV.np_project(X, cams.K[v], cams.dist[v], cams.R[v], cams.t[v])
            for v in range(6)
        ],
        1,
    )
    uv += rng.normal(scale=1.5, size=uv.shape)
    out = rng.random(uv.shape[:2]) < 0.1
    uv[out] += rng.normal(scale=60, size=(out.sum(), 2))
    uv[rng.random(uv.shape[:2]) < 0.2] = np.nan
    tr = MV.robust_triangulate(uv, cams)
    Xs, nv, er = _study_targets(uv, cams.K, cams.dist[:, 0], cams.R, cams.t)
    assert np.array_equal(tr.X, Xs, equal_nan=True)
    assert np.array_equal(tr.nview, nv)
    assert np.array_equal(tr.err, er, equal_nan=True)


def test_cube_centers():
    X = np.full((3, 4, 3), np.nan)
    X[0] = [[0, 0, 0], [2, 0, 0], [9, 9, 9], [1, 1, 1]]
    X[1, 1:] = [[3, 0, 0], [6, 0, 0], [9, 0, 0]]  # centre node 0 missing -> mean
    X[2, :2] = [[1, 1, 1], [2, 2, 2]]  # missing node 0... only 2 finite -> NaN
    X[2, 0] = np.nan
    names = ["a", "b", "c", "d"]
    c = MV.cube_centers(X, names, ["a", "b"])
    assert np.allclose(c[0], [1, 0, 0])
    assert np.allclose(c[1], [6, 0, 0])
    assert np.isnan(c[2]).all()


def test_associate_frame_groups_animals_across_cameras():
    sio_cams = make_cameras(4, distortion=True, seed=6)
    cams = MV.CameraArrays.from_cameras(sio_cams)
    poses = random_poses(3, seed=11, spread=150.0)
    rng = np.random.default_rng(0)
    inst_by_view, truth = {}, {}
    for v, cam in enumerate(sio_cams):
        order = rng.permutation(3)
        pts = np.stack([cv_project(cam, poses[a]) for a in order])
        pts += rng.normal(scale=1.0, size=pts.shape)
        if v == 1:  # a spurious detection and a partly missing animal
            pts = np.concatenate([pts, pts[:1] + 300.0])
            pts[0, :4] = np.nan
        inst_by_view[v] = pts
        truth[v] = {int(a): int(i) for i, a in enumerate(order)}
    animals = MV.associate_frame(cams, inst_by_view, max_animals=3)
    assert len(animals) == 3
    found = []
    for an in animals:
        # which true animal: majority over cameras
        ids = [a for v, i in an.items() for a, ii in truth[v].items() if ii == i]
        assert len(set(ids)) == 1, an
        found.append(ids[0])
        assert len(an) >= 3
    assert sorted(found) == [0, 1, 2]
    # camera 1's spurious detection (index 3) was never used
    assert all(an.get(1) != 3 for an in animals)
    assert MV.associate_frame(cams, inst_by_view, max_animals=1).__len__() == 1


# ---------------------------------------------------------------------------------------
# Floor normal
# ---------------------------------------------------------------------------------------


def test_fit_floor_normal_ransac():
    rng = np.random.default_rng(0)
    n_true = np.array([0.05, -0.04, 1.0])
    n_true /= np.linalg.norm(n_true)
    e1 = np.cross(n_true, [1, 0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n_true, e1)
    pts = rng.uniform(-200, 200, (300, 1)) * e1 + rng.uniform(-200, 200, (300, 1)) * e2
    pts += rng.normal(scale=0.5, size=(300, 1)) * n_true
    pts[:60] += rng.uniform(10, 60, (60, 1)) * n_true  # rearing: 20 % off the floor
    cams = _cams(4)
    n, frac = MV.fit_floor_normal(pts, cams.centers, band=4.0)
    assert np.degrees(np.arccos(n @ n_true)) < 0.5
    assert 0.75 < frac < 0.85
    # pointing to the cameras even when the points say otherwise
    n2, _ = MV.fit_floor_normal(pts, -cams.centers, band=4.0)
    assert np.allclose(n2, -n)


def test_resolve_up_order():
    labels, poses = make_session_labels(n_frames=8, n_animals=2, seed=1)
    session = labels.sessions[0]
    data = MV.extract_session_data(session, NODES)
    rel = data.reliable()
    fit = MV.resolve_up(session, data.cameras, data.X, rel, NODES)
    assert fit.method == "lowest_nodes"
    assert np.degrees(np.arccos(fit.up @ [0, 0, 1])) < 2.0
    fit = MV.resolve_up(
        session, data.cameras, data.X, rel, NODES, up_fit_nodes=["tail_tip"]
    )
    assert fit.method == "up_fit_nodes"
    assert np.degrees(np.arccos(fit.up @ [0, 0, 1])) < 1.0
    session.metadata["up"] = [0.0, 0.1, 1.0]
    fit = MV.resolve_up(
        session, data.cameras, data.X, rel, NODES, up_fit_nodes=["tail_tip"]
    )
    assert fit.method == "metadata" and np.isclose(np.linalg.norm(fit.up), 1)
    fit = MV.resolve_up(session, data.cameras, data.X, rel, NODES, up=[0, 0, 2])
    assert fit.method == "given" and np.allclose(fit.up, [0, 0, 1])


# ---------------------------------------------------------------------------------------
# Session data
# ---------------------------------------------------------------------------------------


def test_extract_session_data_triangulates_groups():
    labels, poses = make_session_labels(n_frames=3, n_animals=2, seed=2)
    data = MV.extract_session_data(labels.sessions[0], NODES)
    assert len(data) == 6
    assert data.frame_idx.tolist() == [0, 0, 1, 1, 2, 2]
    assert data.group_index.tolist() == [0, 1] * 3
    assert data.pts2d.shape == (6, 4, len(NODES), 2)
    assert np.abs(data.X - poses.reshape(6, len(NODES), 3)).max() < 1e-4
    assert (data.nview == 4).all() and not data.given_3d.any()
    assert data.target_mask().all() and data.reliable().all()
    # a node order other than the skeleton's, and a node the skeleton lacks
    d2 = MV.extract_session_data(labels.sessions[0], ["tail_tip", "nose", "ghost"])
    assert np.allclose(d2.X[:, 0], data.X[:, -1]) and np.isnan(d2.X[:, 2]).all()


def test_extract_session_data_prefers_instance_3d_and_user_instances():
    labels, poses = make_session_labels(
        n_frames=2, n_animals=1, seed=3, with_instance_3d=True
    )
    session = labels.sessions[0]
    # move the stored 3D: auto uses it, triangulate ignores it
    ig = session.frame_groups[0].instance_groups[0]
    ig.instance_3d.points = ig.instance_3d.points + 5.0
    auto = MV.extract_session_data(session, NODES)
    assert auto.given_3d.all()
    assert np.allclose(auto.X[0], poses[0, 0] + 5.0)
    assert np.abs(auto.dlt[0] - poses[0, 0]).max() < 1e-4
    assert auto.target_mask().all()
    tri = MV.extract_session_data(session, NODES, target_source="triangulate")
    assert not tri.given_3d.any() and np.abs(tri.X[0] - poses[0, 0]).max() < 1e-4
    # a user instance made from camera 0's prediction wins
    cam0 = session.camera_group.cameras[0]
    lf = session.frame_groups[1].get_frame(cam0)
    pred = session.frame_groups[1].instance_groups[0].get_instance(cam0)
    user = sio.Instance.from_numpy(pred.numpy() + 3.0, skeleton=pred.skeleton)
    user.from_predicted = pred
    lf.instances.append(user)
    d = MV.extract_session_data(session, NODES)
    assert np.allclose(d.pts2d[1, 0], pred.numpy() + 3.0)


def test_session_round_trips_through_slp(tmp_path):
    labels, poses = make_session_labels(
        n_frames=2, n_animals=2, seed=4, with_instance_3d=True
    )
    path = tmp_path / "session.slp"
    labels.save(str(path))
    loaded = sio.load_slp(str(path), open_videos=False)
    a = MV.extract_session_data(labels.sessions[0], NODES)
    b = MV.extract_session_data(loaded.sessions[0], NODES)
    assert np.allclose(a.X, b.X) and np.allclose(a.pts2d, b.pts2d, equal_nan=True)
    assert b.given_3d.all()
    assert np.allclose(a.cameras.K, b.cameras.K) and np.allclose(
        a.cameras.R, b.cameras.R
    )
