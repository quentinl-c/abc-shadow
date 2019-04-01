cdef class Model:
    cdef list _params
    cpdef set_params(self, args)
    cpdef evaluate_from_stats(self, args)
