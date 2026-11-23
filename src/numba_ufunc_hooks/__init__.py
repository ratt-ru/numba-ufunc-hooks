__version__ = "0.0.1"


def _install_extension():
    """Register the fft overloads.

    If NumPy's ufunc layout doesn't match the struct mirror, warn and
    leave the overloads unregistered rather than risk calling through
    a misread pointer.
    """
    import warnings

    from numba_ufunc_hooks.ufunc_minimal import UfuncLayoutError, UfuncLayoutWarning

    try:
        import numba_ufunc_hooks.fft  # noqa: F401
    except UfuncLayoutError as e:
        warnings.warn(
            f"numba-ufunc-hooks is disabled: {e}", UfuncLayoutWarning, stacklevel=2
        )
