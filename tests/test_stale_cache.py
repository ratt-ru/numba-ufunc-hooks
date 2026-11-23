import json
import os
import subprocess
import sys

import numba
import numpy as np
import pytest


@pytest.fixture
def layout_fingerprint():
    """The live layout fingerprint, restored after the test"""
    from numba_ufunc_hooks import fft

    original = fft.LAYOUT_FINGERPRINT.value
    yield fft.LAYOUT_FINGERPRINT
    fft.LAYOUT_FINGERPRINT.value = original


@pytest.mark.parametrize(
    "np_function, dtype",
    [
        (np.fft.fft, np.complex128),
        (np.fft.ifft, np.complex64),
        (np.fft.rfft, np.float64),
        (np.fft.irfft, np.complex128),
    ],
)
def test_stale_cache_raises(layout_fingerprint, np_function, dtype):
    @numba.njit
    def fn(a):
        return np_function(a)

    a = np.ones(8, dtype)
    fn(a)

    # Code compiled under one layout now runs under another,
    # as cached code does after NumPy is upgraded beneath it
    layout_fingerprint.value ^= 1

    with pytest.raises(RuntimeError, match="clear the Numba cache"):
        fn(a)


def test_matching_fingerprint_runs(layout_fingerprint):
    @numba.njit
    def fn(a):
        return np.fft.fft(a)

    a = np.ones(8, np.complex128)
    np.testing.assert_allclose(fn(a), np.fft.fft(a))


CACHED_MODULE = """
import numba
import numpy as np

@numba.njit(cache=True)
def cached_fft(a):
    return np.fft.fft(a)
"""

CALL_CACHED = """
import json
import sys
import warnings

import numpy as np

if sys.argv[1] == "layout-changed":
    from numba_ufunc_hooks import ufunc_minimal

    def broken_guard(ufunc, mirror=ufunc_minimal.MinimalPyUFuncObject):
        raise ufunc_minimal.UfuncLayoutError("forced mismatch")

    ufunc_minimal.validate_mirror = broken_guard

import cached_fft_module

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    try:
        cached_fft_module.cached_fft(np.ones(4, np.complex128))
    except RuntimeError as e:
        outcome = str(e)
    else:
        outcome = "ran"

print(json.dumps({
    "outcome": outcome,
    "warnings": [w.category.__name__ for w in caught],
}))
"""


def test_cache_from_before_a_layout_change_raises(tmp_path):
    """Code cached by one process must fail with a catchable error, not abort
    on an unresolved symbol, in a later process whose NumPy layout no longer
    matches the struct mirror"""
    (tmp_path / "cached_fft_module.py").write_text(CACHED_MODULE)

    def run(mode):
        proc = subprocess.run(
            [sys.executable, "-c", CALL_CACHED, mode],
            capture_output=True,
            text=True,
            cwd=tmp_path,
            env={**os.environ, "PYTHONPATH": str(tmp_path)},
        )
        assert proc.returncode == 0, proc.stderr
        return json.loads(proc.stdout.strip().splitlines()[-1])

    assert run("layout-unchanged")["outcome"] == "ran"
    assert list(tmp_path.glob("__pycache__/*cached_fft*.nbi")), "nothing was cached"

    result = run("layout-changed")
    assert "forced mismatch" in result["outcome"]
    assert "UfuncLayoutWarning" in result["warnings"]
