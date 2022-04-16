from libcpp.vector cimport vector
from libcpp cimport bool

from Public.CppFuns.Keep_order_ cimport Keep_order

from Public.CppFuns.Graph_ cimport Graph

cdef extern from "KSUMS.cpp":
    pass

cdef extern from "KSUMS.h":
    cdef cppclass KSUMS:

        vector[vector[int]] Y
        vector[double] time_arr

        KSUMS() except +
        KSUMS(vector[vector[int]]& NN, vector[vector[double]]& NND, int c_true, bool debug, double max_dd) except +
        void opt(int rep, int ITER, int our_init)
