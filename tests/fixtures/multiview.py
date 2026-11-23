"""A synthetic calibrated rig for the volumetric (``centered_volume``) tests.

Plain helpers (no pytest fixtures): cameras on a ring around the origin, looking at it,
with all five OpenCV distortion terms; animals as random poses above a floor (world
``z`` up); and a ``sio.Labels`` holding a ``RecordingSession`` with frame groups and
instance groups, as LUC3D projects carry them.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import cv2
import numpy as np
import sleap_io as sio

NODES = ["nose", "head", "neck", "trunk", "tail_base", "tail_tip"]
# a stretched-out animal along +x, in world units (mm), centred near "trunk"
_TEMPLATE = np.array(
    [
        [40.0, 0.0, 25.0],
        [28.0, 0.0, 28.0],
        [15.0, 0.0, 25.0],
        [0.0, 0.0, 22.0],
        [-20.0, 0.0, 15.0],
        [-60.0, 0.0, 2.0],
    ]
)


def look_at(position, target=(0.0, 0.0, 0.0), world_up=(0.0, 0.0, 1.0)):
    """``(rvec, tvec)`` of a camera at ``position`` looking at ``target``."""
    c = np.asarray(position, np.float64)
    z = np.asarray(target, np.float64) - c
    z /= np.linalg.norm(z)
    x = np.cross(z, np.asarray(world_up, np.float64))
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])  # rows: camera axes in world coordinates
    t = -R @ c
    rvec, _ = cv2.Rodrigues(R)
    return rvec.ravel(), t


def make_cameras(
    n_views: int = 4,
    radius: float = 700.0,
    height: float = 450.0,
    size=(640, 480),
    focal: float = 600.0,
    distortion: bool = True,
    seed: int = 0,
) -> List[sio.Camera]:
    """Cameras evenly on a ring, all looking at the origin."""
    rng = np.random.default_rng(seed)
    cams = []
    w, h = size
    for i in range(n_views):
        a = 2 * np.pi * i / n_views + 0.3
        pos = (radius * np.cos(a), radius * np.sin(a), height + 40 * (i % 2))
        rvec, tvec = look_at(pos)
        K = np.array(
            [[focal, 0, w / 2 - 0.5], [0, focal * 1.01, h / 2 - 0.5], [0, 0, 1]]
        )
        if distortion:
            dist = np.array(
                [
                    rng.uniform(-0.25, 0.1),
                    rng.uniform(-0.05, 0.05),
                    rng.uniform(-2e-3, 2e-3),
                    rng.uniform(-2e-3, 2e-3),
                    rng.uniform(-0.01, 0.01),
                ]
            )
        else:
            dist = np.zeros(5)
        cams.append(
            sio.Camera(
                matrix=K, dist=dist, size=(w, h), rvec=rvec, tvec=tvec, name=f"cam{i}"
            )
        )
    return cams


def random_poses(n: int, seed: int = 0, spread: float = 120.0) -> np.ndarray:
    """``(n, J, 3)`` animal poses on the floor (``z >= 0``), random place and heading."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        yaw = rng.uniform(0, 2 * np.pi)
        c, s = np.cos(yaw), np.sin(yaw)
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        jitter = rng.normal(scale=2.0, size=_TEMPLATE.shape)
        p = (_TEMPLATE + jitter) @ Rz.T
        p[:, :2] += rng.uniform(-spread, spread, size=2)
        p[:, 2] = np.maximum(p[:, 2], 0.0)
        out.append(p)
    return np.stack(out)


def project(cam: sio.Camera, X: np.ndarray) -> np.ndarray:
    """``(..., 3)`` world points to ``(..., 2)`` pixels with OpenCV."""
    X = np.asarray(X, np.float64)
    uv, _ = cv2.projectPoints(
        X.reshape(-1, 1, 3),
        np.asarray(cam.rvec, np.float64),
        np.asarray(cam.tvec, np.float64),
        np.asarray(cam.matrix, np.float64),
        np.asarray(cam.dist, np.float64),
    )
    return uv.reshape(*X.shape[:-1], 2)


def make_session_labels(
    n_frames: int = 6,
    n_animals: int = 2,
    n_views: int = 4,
    seed: int = 0,
    noise_px: float = 0.0,
    with_instance_3d: bool = False,
    videos: Optional[Sequence[sio.Video]] = None,
    cameras: Optional[List[sio.Camera]] = None,
    poses: Optional[np.ndarray] = None,
):
    """A ``sio.Labels`` with one recording session.

    Every frame has ``n_animals`` animals seen by every camera as predicted instances,
    grouped across cameras into instance groups (``instance_3d`` set to the true pose
    when ``with_instance_3d``).

    Returns:
        A tuple ``(labels, poses)`` with ``poses`` ``(n_frames, n_animals, J, 3)``.
    """
    rng = np.random.default_rng(seed + 1)
    cams = cameras or make_cameras(n_views, seed=seed)
    skel = sio.Skeleton(nodes=list(NODES), name="mouse")
    if videos is None:
        videos = [sio.Video(filename=f"{c.name}.mp4", open_backend=False) for c in cams]
    if poses is None:
        poses = random_poses(n_frames * n_animals, seed=seed).reshape(
            n_frames, n_animals, len(NODES), 3
        )
    session = sio.RecordingSession(camera_group=sio.CameraGroup(cameras=cams))
    for cam, vid in zip(cams, videos):
        session.add_video(vid, cam)
    lfs = []
    for f in range(n_frames):
        groups, lf_by_cam = [], {}
        per_cam = {}
        for cam, vid in zip(cams, videos):
            insts = []
            for a in range(n_animals):
                uv = project(cam, poses[f, a]) + rng.normal(
                    scale=noise_px, size=(len(NODES), 2)
                ) * (noise_px > 0)
                insts.append(
                    sio.PredictedInstance.from_numpy(
                        points_data=uv,
                        point_scores=np.ones(len(NODES)),
                        score=1.0,
                        skeleton=skel,
                    )
                )
            lf = sio.LabeledFrame(video=vid, frame_idx=f, instances=insts)
            lfs.append(lf)
            lf_by_cam[cam] = lf
            per_cam[cam] = insts
        for a in range(n_animals):
            i3 = (
                sio.Instance3D(points=poses[f, a], skeleton=skel)
                if with_instance_3d
                else None
            )
            groups.append(
                sio.InstanceGroup(
                    instance_by_camera={c: per_cam[c][a] for c in cams},
                    instance_3d=i3,
                )
            )
        session.frame_groups[f] = sio.FrameGroup(
            frame_idx=f, instance_groups=groups, labeled_frame_by_camera=lf_by_cam
        )
    labels = sio.Labels(
        labeled_frames=lfs, videos=list(videos), skeletons=[skel], sessions=[session]
    )
    return labels, poses
