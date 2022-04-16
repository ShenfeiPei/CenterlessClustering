from libcpp.vector cimport vector

cdef extern from "KSUMSX.cpp":
    pass

cdef extern from "KSUMSX.h":
    cdef cppclass KSUMSX:

        vector[vector[int]] Y
        vector[double] time_arr
        vector[int] iter_arr

        KSUMSX() except +
        KSUMSX(vector[vector[double]]& X, int c_true, int debug) except +
        void opt(vector[vector[int]]& Y, int b, int ITER)
