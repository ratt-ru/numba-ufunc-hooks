import ctypes

import numpy as np
import pytest
from numpy.fft._pocketfft import pfu

from numba_ufunc_hooks.ufunc_minimal import (
    MinimalPyUFuncObject,
    UfuncLayoutError,
    validate_mirror,
)

POCKETFFT_UFUNCS = ["fft", "ifft", "rfft_n_even", "rfft_n_odd", "irfft"]


class MisalignedPyUFuncObject(ctypes.Structure):
    """A mirror whose ufunc fields sit one pointer too far into the object,
    as they would if the PyObject header grew"""

    _fields_ = [
        MinimalPyUFuncObject._fields_[0],
        MinimalPyUFuncObject._fields_[1],
        ("unexpected", ctypes.c_void_p),
        *MinimalPyUFuncObject._fields_[2:],
    ]


@pytest.mark.parametrize("name", POCKETFFT_UFUNCS)
def test_mirror_matches_pocketfft_ufuncs(name):
    validate_mirror(getattr(pfu, name))


def test_mirror_matches_plain_ufunc():
    validate_mirror(np.add)


@pytest.mark.parametrize("name", POCKETFFT_UFUNCS)
def test_misaligned_mirror_is_rejected(name):
    with pytest.raises(UfuncLayoutError, match=name):
        validate_mirror(getattr(pfu, name), MisalignedPyUFuncObject)
