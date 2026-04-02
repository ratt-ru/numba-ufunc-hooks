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
