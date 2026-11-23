import ctypes
from enum import IntEnum

import numpy as np


class CTypesEnum(IntEnum):
    """A ctypes-compatible enumeration base class"""

    @classmethod
    def from_param(cls, obj) -> int:
        return int(obj)


NPY_SAME_VALUE_CASTING_FLAG = 64


class NPY_CASTING(CTypesEnum):
    _NPY_ERROR_OCCURRED_IN_CAST = -1
    NPY_NO_CASTING = 0
    NPY_EQUIV_CASTING = 1
    NPY_SAFE_CASTING = 2
    NPY_SAME_KIND_CASTING = 3
    NPY_UNSAFE_CASTING = 4
    NPY_SAME_VALUE_CASTING = NPY_UNSAFE_CASTING | NPY_SAME_VALUE_CASTING_FLAG


# Numpy types expressed as ctypes
_CTYPES_NPY_INTP = np.ctypeslib.as_ctypes_type(np.intp)
_CTYPES_NPY_UINT32 = np.ctypeslib.as_ctypes_type(np.uint32)

# The PyObject header differs between builds: the default build has
# (ob_refcnt, ob_type), while the free-threaded build has a larger header
# (ob_tid, flags, mutex, gc bits, local and shared refcounts, ob_type).
# Both end in ob_type, and sizeof(PyObject) is object.__basicsize__,
# so everything before ob_type is mirrored as opaque bytes.
_PYOBJECT_HEAD_SIZE = object.__basicsize__ - ctypes.sizeof(ctypes.c_void_p)

# ctypes function signature for a UFunc
PyUFuncGenericFunction = ctypes.CFUNCTYPE(
    # return type: void
    None,
    # char ** args
    ctypes.POINTER(ctypes.c_char_p),
    # npy_intp * dimensions
    ctypes.POINTER(_CTYPES_NPY_INTP),
    # npy_intp * steps
    ctypes.POINTER(_CTYPES_NPY_INTP),
    # void * data
    ctypes.c_void_p,
)

# ctypes signature for TypeResolutionFunc
# TODO: some arguments depend on undefined types
PyUFunc_TypeResolutionFunc = ctypes.CFUNCTYPE(
    # return type
    ctypes.c_int,
    # ufunc
    ctypes.POINTER(PyUFuncGenericFunction),
    # casting: NPY_CASTING
    ctypes.c_int,
    # operands: PyArrayObject
    ctypes.POINTER(ctypes.c_void_p),
    # type_tup: PyObject
    ctypes.c_void_p,
    # out_dtype: PyArray_Descr
    ctypes.POINTER(ctypes.c_void_p),
)


class MinimalPyUFuncObject(ctypes.Structure):
    """This structure replicates a subset of the following numpy struct

    https://numpy.org/devdocs/reference/c-api/types-and-structures.html#c.PyUFuncObject
    https://github.com/numpy/numpy/blob/main/numpy/_core/include/numpy/ufuncobject.h#L102
    """

    _fields_ = [
        # PyObject members
        ("ob_head", ctypes.c_char * _PYOBJECT_HEAD_SIZE),
        ("ob_type", ctypes.c_void_p),
        # PyUfuncObject members follow
        ("nin", ctypes.c_int),
        ("nout", ctypes.c_int),
        ("nargs", ctypes.c_int),
        ("identity", ctypes.c_int),
        ("functions", ctypes.POINTER(PyUFuncGenericFunction)),
        ("data", ctypes.POINTER(ctypes.c_void_p)),
        ("ntypes", ctypes.c_int),
        ("reserved1", ctypes.c_int),
        ("name", ctypes.c_char_p),
        # Type numbers, ntypes * nargs of them. Not a C string:
        # NPY_BOOL is 0, so a c_char_p would stop early
        ("types", ctypes.POINTER(ctypes.c_ubyte)),
        ("doc", ctypes.c_char_p),
        ("ptr", ctypes.c_void_p),
        # PyObject
        ("obj", ctypes.c_void_p),
        # PyObject
        ("userloops", ctypes.c_void_p),
        # 0 for scalar ufunc; 1 for generalized ufunc
        ("core_enabled", ctypes.c_int),
        # Number of distinct dimension names in signature
        ("core_num_dim_ix", ctypes.c_int),
        # dimension indices of input/output argument k are stored in
        # core_dim_ixs[core_offsets[k]..core_offsets[k]+core_num_dims[k]-1]``
        #
        # number of core dimensions of each argument
        ("core_num_dims", ctypes.POINTER(ctypes.c_int)),
        # dimension indices in a flatted form; indices
        # are in the range of [0,core_num_dim_ix)
        ("core_dim_ixs", ctypes.POINTER(ctypes.c_int)),
        # positions of 1st core dimensions of each
        # argument in core_dim_ixs, equivalent to cumsum(core_num_dims)
        ("core_offsets", ctypes.POINTER(ctypes.c_int)),
        # signature string for printing purposes
        ("core_signature", ctypes.c_char_p),
        # A function which resolves the types and fills an array
        # with the dtypes for inputs and outputs.
        ("type_resolver", PyUFunc_TypeResolutionFunc),
        # A dictionary for monkey patching ufuncs
        ("dict", ctypes.c_void_p),
        # Unused inner loop selector
        ("vectorcall", ctypes.c_void_p),
        # Was previously the `PyUFunc_MaskedInnerLoopSelectionFunc`
        ("reserved3", ctypes.c_void_p),
        # List of flags for each operand when ufunc
        # is called by the nditer object
        ("op_flags", ctypes.POINTER(_CTYPES_NPY_UINT32)),
        # List of global flags used when ufunc is called by nditer object.
        ("iter_flags", _CTYPES_NPY_UINT32),
        ("core_dim_sizes", ctypes.POINTER(_CTYPES_NPY_INTP)),
        ("core_dim_flags", ctypes.POINTER(_CTYPES_NPY_UINT32)),
        # More PyUfuncObject members can follow, but
        # the above are sufficient for extracting ufunc
        # function pointers
    ]


class UfuncLayoutError(RuntimeError):
    """The struct mirror disagrees with the running NumPy's PyUFuncObject"""


class UfuncLayoutWarning(RuntimeWarning):
    """Issued when the hooks are skipped because of a UfuncLayoutError"""


def validate_mirror(ufunc: np.ufunc, mirror=MinimalPyUFuncObject) -> None:
    """Check that ``mirror`` reads ``ufunc`` the way Python itself reports it.

    Every field Python exposes is compared with what the mirror reads at
    that field's offset. The checks run from the start of the struct
    outwards, and pointers are only followed once every plain integer
    field has matched, so a badly misaligned mirror is rejected before
    it dereferences garbage.

    Raises:
        UfuncLayoutError: listing each field that disagrees.
    """
    name = ufunc.__name__

    def fail(problems):
        raise UfuncLayoutError(
            f"NumPy's PyUFuncObject layout for '{name}' does not match the "
            f"struct mirror (NumPy {np.__version__}): {'; '.join(problems)}"
        )

    basicsize = type(ufunc).__basicsize__
    if ctypes.sizeof(mirror) > basicsize:
        fail([f"mirror is {ctypes.sizeof(mirror)} bytes, ufunc is {basicsize}"])

    view = mirror.from_address(id(ufunc))
    problems = []

    if view.ob_type != id(type(ufunc)):
        problems.append("ob_type is not type(ufunc)")

    for field in ("nin", "nout", "nargs", "ntypes"):
        if (got := getattr(view, field)) != (want := getattr(ufunc, field)):
            problems.append(f"{field} is {got}, expected {want}")

    want_core = ufunc.signature is not None
    if view.core_enabled != want_core:
        problems.append(f"core_enabled is {view.core_enabled}, expected {want_core}")

    if problems:
        fail(problems)

    # The integer fields agree, so the pointers should be safe to follow
    if (got_name := view.name) != name.encode():
        problems.append(f"name is {got_name!r}")

    if not view.functions:
        problems.append("functions is NULL")

    if not view.types:
        problems.append("types is NULL")
    else:
        want_types = [
            np.dtype(c).num for sig in ufunc.types for c in sig.replace("->", "")
        ]
        got_types = view.types[: len(want_types)]
        if got_types != want_types:
            problems.append(f"types are {got_types}, expected {want_types}")

    want_sig = None if ufunc.signature is None else ufunc.signature.encode()
    # NumPy stores the signature as given and Python reports it
    # with whitespace removed
    got_sig = view.core_signature
    if got_sig is not None:
        got_sig = got_sig.replace(b" ", b"")
    if got_sig != want_sig:
        problems.append(f"core_signature is {got_sig!r}, expected {want_sig!r}")

    if problems:
        fail(problems)
