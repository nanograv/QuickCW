"""C 2021 Matthew Digman
various jit compatible interfaces to cython lapack functions """
import ctypes
from numba.extending import get_cython_function_address
from numba import njit
import numpy as np

_PTR  = ctypes.POINTER

_dble = ctypes.c_double
_char = ctypes.c_char
_int  = ctypes.c_int

_ptr_select = ctypes.c_voidp
_ptr_dble = _PTR(_dble)
_ptr_char = _PTR(_char)
_ptr_int  = _PTR(_int)


# signature is:
# void dtrtrs(
#  char *UPLO,
#  char *TRANS,
#  char *DIAG,
#  int *N,
#  int *NRHS,
#  d *A,
#  int *LDA,
#  d *B,
#  int *LDB,
#  int *info
# )
# bind to the real space variant of the function
addr = get_cython_function_address('scipy.linalg.cython_lapack', 'dtrtrs')
functype = ctypes.CFUNCTYPE(None,
                            _ptr_int, # UPLO
                            _ptr_int, # TRANS
                            _ptr_int, # DIAG
                            _ptr_int, # N
                            _ptr_int, # NRHS
                            _ptr_dble, # A
                            _ptr_int, # LDA
                            _ptr_dble, # B
                            _ptr_int, # LDB
                            _ptr_int, # INFO
                            )
dtrtrs_fn = functype(addr)
@njit()
def solve_triangular(x,y,lower_a=True,trans_a=True,unitdiag=False,overwrite_b=False):
    """solve x*B=y

    :param x:   triangular matrix (must be either type of contiguous)
    :param y:   vector (must be fortran ordered)
    
    :return B:  Solution to x*B=y
    """
    #if the input matrix is c contiguous but not fortran contiguous
    #transposing it will make it fortran contiguous with no copying
    #then flipping upper and lower and telling dtrtrs to undo the transpose will force dtrtrs to do the correct operation
    if x.flags.c_contiguous and not x.flags.f_contiguous:
        trans_a = not trans_a
        lower_a = not lower_a
        A = x.T
    else:
        A = x     # in & out

    if trans_a:
        TRANS = np.array([ord('T')], np.int32)
    else:
        TRANS = np.array([ord('N')], np.int32)

    if lower_a:
        UPLO = np.array([ord('L')], np.int32)
    else:
        UPLO = np.array([ord('U')], np.int32)

    #TODO why was this in place? added copy to mitigate, ensure nothing relied on that behavior
    if overwrite_b:
        B = y
    else:
        B = y.T.copy().T

    #cannot do this operation in place if y is not contiguous, though could copy
    if not (A.flags.f_contiguous and B.flags.f_contiguous):
        raise ValueError('x must be contiguous and y must be fortran contiguous')


    if unitdiag:
        DIAG = np.array([ord('U')], np.int32)
    else:
        DIAG = np.array([ord('N')], np.int32)

    _M, _N = x.shape
    if y.ndim==1:
        _LDB = y.size
        _NB = 1
    else:
        _LDB,_NB = y.shape
    if _LDB != _N or _M!=_N:
        raise ValueError('x must be square and y must have same first dimension as x')

    N = np.array(_N, np.int32)
    NRHS = np.array(_NB, np.int32)
    LDA = np.array(_N, np.int32)
    LDB = np.array(_LDB, np.int32) #changed from _N

    INFO = np.empty(1, dtype=np.int32)

    def check_info(info):
        if info[0] != 0:
            print(info)
            raise RuntimeError("INFO indicates problem with dtrtrs")

    dtrtrs_fn(UPLO.ctypes,
             TRANS.ctypes,
             DIAG.ctypes,
             N.ctypes,
             NRHS.ctypes,
             A.ctypes,
             LDA.ctypes,
             B.ctypes,
             LDB.ctypes,
             INFO.ctypes)

    check_info(INFO)
    return B
