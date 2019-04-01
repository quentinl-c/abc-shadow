from model cimport Model
from libcpp.vector cimport vector as cpp_vector
from libcpp.utility cimport pair as cpp_pair

ctypedef cpp_pair[int, int] cpp_edge

cdef class GraphModel(Model):
    cpdef double compute_delta(self, mut_sample, cpp_edge edge, int new_val)
    cdef double _compute_delta(self, mut_sample, cpp_edge edge, int new_val)