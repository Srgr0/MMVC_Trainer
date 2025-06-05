cimport cython
from cython.parallel import prange
import numpy as np
cimport numpy as np


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void maximum_path_each(int[:,::1] path, float[:,::1] value, int t_y, int t_x, float max_neg_val=-1e9) nogil:
    """
    Single sequence maximum path search.
    
    Args:
        path: Output path matrix [t_y, t_x]
        value: Negative cost matrix [t_y, t_x] 
        t_y: Valid length in y dimension (text/phoneme)
        t_x: Valid length in x dimension (spectrogram)
        max_neg_val: Maximum negative value for initialization
    """
    cdef int x, y
    cdef float v_prev, v_cur
    cdef int index = t_x - 1

    # Forward pass - dynamic programming
    for y in range(t_y):
        for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
            if x == y:
                v_cur = max_neg_val
            else:
                v_cur = value[y-1, x]
            
            if x == 0:
                if y == 0:
                    v_prev = 0.0
                else:
                    v_prev = max_neg_val
            else:
                v_prev = value[y-1, x-1]
            
            value[y, x] += max(v_prev, v_cur)

    # Backward pass - backtracking
    for y in range(t_y - 1, -1, -1):
        path[y, index] = 1
        if index != 0 and (index == y or value[y-1, index] < value[y-1, index-1]):
            index = index - 1


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void maximum_path_c(int[:,:,::1] paths, float[:,:,::1] values, int[::1] t_ys, int[::1] t_xs) nogil:
    """
    Batch maximum path search with parallel processing.
    
    Args:
        paths: Output paths [batch, t_y, t_x]
        values: Negative cost matrices [batch, t_y, t_x]
        t_ys: Valid lengths in y dimension [batch]
        t_xs: Valid lengths in x dimension [batch]
    """
    cdef int b = paths.shape[0]
    cdef int i
    
    for i in prange(b, nogil=True):
        maximum_path_each(paths[i], values[i], t_ys[i], t_xs[i])
