Numba hooks to NumPy ufuncs
===========================

This repository contains some hooks to NumPy ufuncs. Currently it provides overloads for:

- ``np.fft.fft``
- ``np.fft.ifft``
- ``np.fft.rfft``
- ``np.fft.irfft``

allowing the following in a numba kernel


.. code-block:: python

  @numba.njit(nogil=True)
  def fn(a):
    return np.fft.fft(a)

Troubleshooting
---------------

``UfuncLayoutWarning`` on import
  This version of numba-ufunc-hooks doesn't support your NumPy build, and
  the FFT overloads aren't available. Calling ``np.fft.*`` from a jitted
  function fails to compile. Please open an issue and include your NumPy
  and Python versions.

``RuntimeError`` asking you to clear the Numba cache
  A function compiled with ``cache=True`` was cached under a different
  NumPy version. Delete the ``.nbi`` and ``.nbc`` files in the
  ``__pycache__`` directories next to your code, or in ``NUMBA_CACHE_DIR``
  if you set it, and run again.
