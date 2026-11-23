import json
import subprocess
import sys

# Runs in a fresh interpreter so that the numba_extensions entry point
# loads the package for the first time with the ABI guard forced to fail.
SCRIPT = """
import json
import warnings

import numba
import numpy as np

from numba_ufunc_hooks import ufunc_minimal

def broken_guard(ufunc, mirror=ufunc_minimal.MinimalPyUFuncObject):
    raise ufunc_minimal.UfuncLayoutError("forced mismatch")

ufunc_minimal.validate_mirror = broken_guard

@numba.njit
def fn(a):
    return np.fft.fft(a)

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    try:
        fn(np.ones(4, np.complex128))
    except numba.core.errors.TypingError:
        compiled = False
    else:
        compiled = True

print(json.dumps({
    "compiled": compiled,
    "warnings": [(w.category.__name__, str(w.message)) for w in caught],
}))
"""


def test_layout_mismatch_warns_and_skips_fft_overloads():
    proc = subprocess.run(
        [sys.executable, "-c", SCRIPT], capture_output=True, text=True, check=True
    )
    result = json.loads(proc.stdout.strip().splitlines()[-1])

    assert not result["compiled"], "fft overloads were registered despite mismatch"
    layout_warnings = [
        msg for cat, msg in result["warnings"] if cat == "UfuncLayoutWarning"
    ]
    assert len(layout_warnings) == 1
    assert "forced mismatch" in layout_warnings[0]
