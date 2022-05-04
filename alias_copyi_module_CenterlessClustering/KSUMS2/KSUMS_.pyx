# distutils: language = c++

cimport numpy as np
import numpy as np
np.import_array()

from KSUMS_ cimport KSUMS

cdef class PyKSUMS:
    cdef KSUMS c_KSUMS

    def __cinit__(self, np.ndarray[int, ndim=2] NN, np.ndarray[double, ndim=2] NND, c_true, debug, max_dd):
        self.c_KSUMS = KSUMS(NN, NND, c_true, debug, max_dd)

    def opt(self, rep, ITER, our_init):
        self.c_KSUMS.opt(rep, ITER, our_init)

    @property
    def y_pre(self):
        return np.array(self.c_KSUMS.Y)

    @property
    def times(self):
        return np.array(self.c_KSUMS.time_arr)
