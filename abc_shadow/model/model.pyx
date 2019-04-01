import numpy as np

cdef class Model:

    def __init__(self, *params):
        self._params = list(params)

    cpdef set_params(self, args):
        self._params = list(args)

    cpdef evaluate_from_stats(self, args):
        return np.dot(self._params, args)

    @staticmethod
    def summary(results):
        pass

    def evaluate(self, sample):
        pass
