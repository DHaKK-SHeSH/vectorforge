import numpy as np

from vectorforge.io import read_fvecs, write_fvecs


def test_fvecs_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((50, 16)).astype(np.float32)
    path = tmp_path / "x.fvecs"
    write_fvecs(path, x)
    y = read_fvecs(path)
    assert y.shape == (50, 16)
    assert y.dtype == np.float32
    np.testing.assert_array_equal(x, y)
