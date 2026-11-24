"""Read and write LUC3D's volume 3D file (``.vol3d.h5``, format ``lucid-volume3d/1``).

A volumetric model gives, per frame, animal and node, a 3D mean and a covariance (the
second moments of its softmax over the cube). LUC3D (File > Load Volume 3D) gates each
camera's 2D detection by the volume's projection and fuses its triangulation with the
volume by their uncertainties. This module writes the file it reads, with the same code
as LUC3D's ``scripts/export_volume3d.py`` (talmolab/luc3d#510), so the two cannot drift:

    OUT.vol3d.h5
      /frame_indices  int64   (R,)       the session's frame index of each row
                                         (one row per animal per frame)
      /means          float32 (R, J, 3)  world units (the calibration's)
      /covs           float32 (R, J, 6)  [xx, xy, xz, yy, yz, zz], as the model gives them
      /node_names     str     (J,)
      attrs: format = "lucid-volume3d/1", cov_scale, source,
             skeleton_chains (optional JSON)

``cov_scale`` is the factor LUC3D multiplies the covariance by before fusing (its gate
uses the raw covariance): a softmax's spread is much wider than the model's real error
(17-25 times in variance). It is measured where triangulation is good: on points with
at least 4 views and below 3 px, the median squared Mahalanobis distance of
``mean - DLT`` under the covariance, divided by the chi-square (3) median 2.366.

Rows carry no identity: LUC3D matches each row to a camera group by 3D distance.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

FORMAT = "lucid-volume3d/1"
CHI2_3_MEDIAN = 2.366
# cov_scale points: robust DLT from at least this many views, below this error (px)
COV_SCALE_MIN_VIEWS = 4
COV_SCALE_MAX_PX = 3.0
COV_SCALE_MIN_POINTS = 20


def chain_nodes(chain: Dict[str, Any]) -> set:
    """Every node a skeleton chain names (its bones' ends and its nodes)."""
    return {n for b in chain["bones"] for n in b} | set(chain["nodes"])


def load_chains(path: str, node_names: Sequence[str]) -> List[Dict[str, Any]]:
    """Read skeleton chains for LUC3D's skeleton check from a JSON file.

    The file is a list of ``{"name", "bones": [[a, b], ...], "nodes": [...]}``:
    ``bones`` are length-checked, ``nodes`` may be replaced.

    Args:
        path: The JSON file.
        node_names: The file's nodes; every chain node must be one of them.

    Returns:
        The chains.

    Raises:
        ValueError: On a malformed file or an unknown node.
    """
    with open(path) as f:
        chains = json.load(f)
    if not isinstance(chains, list):
        raise ValueError(f"{path}: expected a JSON list of {{name, bones, nodes}}.")
    have = set(node_names)
    for ch in chains:
        if not isinstance(ch, dict) or not ch.get("bones") or not ch.get("nodes"):
            name = ch.get("name") if isinstance(ch, dict) else ch
            raise ValueError(f"{path}: chain {name!r} needs bones and nodes.")
        missing = chain_nodes(ch) - have
        if missing:
            raise ValueError(
                f"{path}: chain {ch.get('name')!r} names nodes not in the file: "
                f"{sorted(missing)}."
            )
    return chains


def pack_covariances(covs: np.ndarray) -> np.ndarray:
    """``(..., 3, 3)`` covariances to ``(..., 6)`` ``[xx, xy, xz, yy, yz, zz]``."""
    c = np.asarray(covs)
    return np.stack(
        [
            c[..., 0, 0],
            c[..., 0, 1],
            c[..., 0, 2],
            c[..., 1, 1],
            c[..., 1, 2],
            c[..., 2, 2],
        ],
        -1,
    )


def unpack_covariances(c6: np.ndarray) -> np.ndarray:
    """``(..., 6)`` packed covariances back to symmetric ``(..., 3, 3)``."""
    c6 = np.asarray(c6)
    xx, xy, xz, yy, yz, zz = (c6[..., i] for i in range(6))
    return np.stack(
        [
            np.stack([xx, xy, xz], -1),
            np.stack([xy, yy, yz], -1),
            np.stack([xz, yz, zz], -1),
        ],
        -2,
    )


def to_rows(
    frames: np.ndarray, means: np.ndarray, covs: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Frame-by-animal arrays to the file's rows.

    Args:
        frames: ``(F,)`` frame indices.
        means: ``(F, A, J, 3)`` means (NaN = no animal / no node).
        covs: ``(F, A, J, 3, 3)`` covariances.

    Returns:
        A tuple ``(frame_indices (R,), means (R, J, 3), covs (R, J, 6))`` over the
        animals with any finite node.
    """
    keep = np.isfinite(means).all(-1).any(-1)  # (F, A)
    fi, ai = np.nonzero(keep)
    m = means[fi, ai].astype(np.float32)
    c6 = pack_covariances(covs[fi, ai])
    return np.asarray(frames)[fi].astype(np.int64), m, c6.astype(np.float32)


def measure_cov_scale(
    means: np.ndarray,
    covs: np.ndarray,
    X: np.ndarray,
    nview: np.ndarray,
    err: np.ndarray,
    min_views: int = COV_SCALE_MIN_VIEWS,
    max_px: float = COV_SCALE_MAX_PX,
) -> Tuple[Optional[float], int]:
    """Measure ``cov_scale`` against a robust DLT, as LUC3D's exporter does.

    Args:
        means: ``(..., 3)`` volume means.
        covs: ``(..., 3, 3)`` volume covariances (raw).
        X: ``(..., 3)`` robust DLT of the same points.
        nview: ``(...)`` the DLT's views.
        err: ``(...)`` its mean reprojection error (px).
        min_views: Views a point needs.
        max_px: Error a point must stay below.

    Returns:
        A tuple ``(cov_scale, n_points)``; ``cov_scale`` is ``None`` with fewer than
        20 such points.
    """
    means = np.asarray(means, np.float64)
    covs = np.asarray(covs, np.float64)
    X = np.asarray(X, np.float64)
    with np.errstate(invalid="ignore"):
        ok = (
            np.isfinite(means).all(-1)
            & np.isfinite(X).all(-1)
            & (np.asarray(nview) >= min_views)
            & (np.asarray(err) < max_px)
            & np.isfinite(covs).all((-1, -2))
        )
    n = int(ok.sum())
    if n < COV_SCALE_MIN_POINTS:
        return None, n
    dd = (means - X)[ok]
    m2 = np.einsum("mi,mij,mj->m", dd, np.linalg.inv(covs[ok] + 1e-6 * np.eye(3)), dd)
    return float(np.median(m2) / CHI2_3_MEDIAN), n


def write_volume3d(
    path: str,
    frame_indices: np.ndarray,
    means: np.ndarray,
    covs6: np.ndarray,
    node_names: Sequence[str],
    cov_scale: float,
    source: str,
    chains: Sequence[Dict[str, Any]] = (),
):
    """Write a ``.vol3d.h5`` file.

    Args:
        path: Output path.
        frame_indices: ``(R,)`` int64.
        means: ``(R, J, 3)`` float32.
        covs6: ``(R, J, 6)`` float32 packed covariances.
        node_names: ``J`` names.
        cov_scale: The fusion scale (see the module docstring).
        source: What made the file.
        chains: Optional skeleton chains (written only when given).
    """
    import h5py

    with h5py.File(path, "w") as h:
        h.create_dataset("frame_indices", data=np.asarray(frame_indices, np.int64))
        h.create_dataset("means", data=np.asarray(means, np.float32))
        h.create_dataset("covs", data=np.asarray(covs6, np.float32))
        h.create_dataset(
            "node_names", data=np.array(list(node_names), dtype=h5py.string_dtype())
        )
        h.attrs["format"] = FORMAT
        h.attrs["cov_scale"] = float(cov_scale)
        h.attrs["source"] = source
        if chains:
            h.attrs["skeleton_chains"] = json.dumps(list(chains))


def read_volume3d(path: str) -> Dict[str, Any]:
    """Read a ``.vol3d.h5`` file.

    Returns:
        A dict with ``frame_indices``, ``means``, ``covs`` (packed ``(R, J, 6)``),
        ``node_names``, ``cov_scale``, ``source``, ``format`` and ``skeleton_chains``
        (``None`` when absent).

    Raises:
        ValueError: When the file is not ``lucid-volume3d/1``.
    """
    import h5py

    with h5py.File(path, "r") as h:
        fmt = h.attrs.get("format")
        fmt = fmt.decode() if isinstance(fmt, bytes) else fmt
        if fmt != FORMAT:
            raise ValueError(f"{path}: format {fmt!r}, expected {FORMAT!r}.")
        names = [
            n.decode() if isinstance(n, bytes) else str(n) for n in h["node_names"][:]
        ]
        chains = h.attrs.get("skeleton_chains")
        src = h.attrs.get("source", "")
        return dict(
            frame_indices=h["frame_indices"][:],
            means=h["means"][:],
            covs=h["covs"][:],
            node_names=names,
            cov_scale=float(h.attrs["cov_scale"]),
            source=src.decode() if isinstance(src, bytes) else str(src),
            format=fmt,
            skeleton_chains=json.loads(chains) if chains is not None else None,
        )
