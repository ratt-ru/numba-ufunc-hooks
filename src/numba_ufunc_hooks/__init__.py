__version__ = "0.0.0"


def _install_extension():
    """Register the fft overloads"""
    import numba_ufunc_hooks.fft  # noqa: F401
