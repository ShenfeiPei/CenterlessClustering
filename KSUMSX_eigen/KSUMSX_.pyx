# distutils: language = c++

cimport numpy as np
import numpy as np
np.import_array()

from KSUMSX_ cimport KSUMSX

cdef class PyKSUMSX:
    cdef KSUMSX c_KSUMSX

    def __cinit__(self, np.ndarray[double, ndim=2] X, int c_true, int debug):
        self.c_KSUMSX = KSUMSX(X, c_true, debug)

    def opt(self, np.ndarray[int, ndim=2] Y, int b, int ITER):
        self.c_KSUMSX.opt(Y, b, ITER)


    @property
    def y_pre(self):
        ret = self.c_KSUMSX.Y
        return np.array(ret)

    @property
    def times(self):
        ret = self.c_KSUMSX.time_arr
        return np.array(ret)

    @property
    def iters(self):
        ret = self.c_KSUMSX.iter_arr
        return np.array(ret)

