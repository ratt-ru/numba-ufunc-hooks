Changelog
=========

X.Y.Z (DD-MM-YYYY)
------------------
* Issue a ``UfuncLayoutWarning`` and leave the FFT overloads unregistered
  on unsupported NumPy builds
* Support free-threaded Python
* Raise a ``RuntimeError`` asking for the Numba cache to be cleared when
  cached code was compiled under a different NumPy version
* Test against the oldest supported NumPy, free-threaded Python 3.14t,
  and nightly NumPy on a schedule
* Retest in github actions to ensure caching works (:pr:`2`)

0.0.1 (02-04-2026)
------------------
* Initial release
* Add github actions and pre-commit hooks (:pr:`1`)
