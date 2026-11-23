"""Tests for LUC3D's `.vol3d.h5` writer and reader (`sleap_nn.io.volume3d`).

The first cases are LUC3D's own `scripts/export_volume3d.py --self-test` (talmolab/luc3d
#510), so the two writers stay in step.
"""

import json

import h5py
import numpy as np
import pytest

from sleap_nn.io import volume3d as V3D


def _data():
    rng = np.random.default_rng(0)
    F, A, J = 4, 2, 3
    means = rng.normal(size=(F, A, J, 3))
    means[1, 1] = np.nan  # an animal missing on a frame
    L = rng.normal(size=(F, A, J, 3, 3))
    covs = L @ np.swapaxes(L, -1, -2) + np.eye(3)
    return means, covs


def test_to_rows_and_packing():
    means, covs = _data()
    F, A = means.shape[:2]
    fi, m, c6 = V3D.to_rows(np.arange(10, 10 + F), means, covs)
    assert len(fi) == F * A - 1 and fi[0] == 10 and fi.dtype == np.int64
    assert m.dtype == np.float32 and c6.dtype == np.float32 and c6.shape[-1] == 6
    c = covs[0, 0, 0]
    assert np.allclose(c6[0, 0], [c[0, 0], c[0, 1], c[0, 2], c[1, 1], c[1, 2], c[2, 2]])
    assert np.allclose(V3D.unpack_covariances(c6[0, 0]), c, atol=1e-5)


def test_measure_cov_scale():
    """A volume whose spread is 4x its real error in sd gives 1/16 in variance."""
    rng = np.random.default_rng(0)
    n = 4000
    X = rng.normal(size=(n, 3)) * 10
    S = np.tile(np.eye(3) * 16.0, (n, 1, 1))
    Vm = X + rng.normal(size=X.shape)
    s, cnt = V3D.measure_cov_scale(Vm, S, X, np.full(n, 5), np.full(n, 1.0))
    assert abs(s - 1 / 16) < 0.01 and cnt == n
    # the exporter's thresholds: >= 4 views and < 3 px
    nv = np.full(n, 5)
    nv[: n // 2] = 3
    er = np.full(n, 1.0)
    er[n // 2 : n // 2 + 100] = 3.0
    _, cnt = V3D.measure_cov_scale(Vm, S, X, nv, er)
    assert cnt == n // 2 - 100
    # too few points: no value
    s, cnt = V3D.measure_cov_scale(Vm[:19], S[:19], X[:19], nv[-19:], er[-19:])
    assert s is None and cnt == 19


def test_write_read_round_trip(tmp_path):
    means, covs = _data()
    fi, m, c6 = V3D.to_rows(np.arange(10, 14), means, covs)
    out = tmp_path / "t.vol3d.h5"
    ch = [{"name": "ab", "bones": [["a", "b"]], "nodes": ["b"]}]
    V3D.write_volume3d(str(out), fi, m, c6, ["a", "b", "c"], 0.05, "test", ch)
    with h5py.File(out, "r") as h:  # the layout LUC3D's reader expects
        assert set(h.keys()) == {"frame_indices", "means", "covs", "node_names"}
        assert h["means"].shape == (7, 3, 3) and h["means"].dtype == np.float32
        assert h["covs"].shape == (7, 3, 6) and h["frame_indices"].dtype == np.int64
        assert h.attrs["format"] == "lucid-volume3d/1"
        assert json.loads(h.attrs["skeleton_chains"]) == ch
    r = V3D.read_volume3d(str(out))
    assert r["node_names"] == ["a", "b", "c"] and r["cov_scale"] == pytest.approx(0.05)
    assert np.array_equal(r["frame_indices"], fi) and np.array_equal(r["means"], m)
    assert r["skeleton_chains"] == ch and r["source"] == "test"
    V3D.write_volume3d(str(out), fi, m, c6, ["a", "b", "c"], 0.05, "test")
    with h5py.File(out, "r") as h:
        assert "skeleton_chains" not in h.attrs
    assert V3D.read_volume3d(str(out))["skeleton_chains"] is None


def test_read_refuses_other_formats(tmp_path):
    out = tmp_path / "x.h5"
    with h5py.File(out, "w") as h:
        h.attrs["format"] = "something-else"
    with pytest.raises(ValueError, match="lucid-volume3d/1"):
        V3D.read_volume3d(str(out))


def test_load_chains(tmp_path):
    names = ["TTI", "Tail_0", "Tail_1", "TailTip"]
    good = [
        {
            "name": "tail",
            "bones": [["TTI", "Tail_0"], ["Tail_0", "Tail_1"], ["Tail_1", "TailTip"]],
            "nodes": ["Tail_0", "Tail_1", "TailTip"],
        }
    ]
    p = tmp_path / "chains.json"
    p.write_text(json.dumps(good))
    assert V3D.load_chains(str(p), names) == good
    p.write_text(json.dumps(good))
    with pytest.raises(ValueError, match="not in the file"):
        V3D.load_chains(str(p), names[:-1])
    p.write_text(json.dumps([{"name": "x", "bones": []}]))
    with pytest.raises(ValueError, match="needs bones and nodes"):
        V3D.load_chains(str(p), names)
    p.write_text(json.dumps({"name": "x"}))
    with pytest.raises(ValueError, match="JSON list"):
        V3D.load_chains(str(p), names)
