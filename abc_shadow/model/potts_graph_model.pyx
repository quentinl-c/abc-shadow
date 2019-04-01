# distutils: language = c++
# cython: boundscheck = False

from graph_model cimport GraphModel
import numpy as np
cimport numpy as np
from collections import Iterable

from numpy cimport ndarray

cdef class PottsGraphModel(GraphModel):
    type_values = {0, 1, 2}

    def __init__(self, *args):
        if len(args) != 3:
            raise ValueError

        super().__init__(*args)

    @property
    def beta01(self):
        return self._params[0]

    @property
    def beta02(self):
        return self._params[1]

    @property
    def beata12(self):
        return self._params[2]

    def set_params(self, *args):

        if len(args) != 3:
            raise ValueError
        super().set_params(*args)

    def evaluate_from_stats(self, *args):

        if len(args) != 3:
            raise ValueError

        return super().evaluate_from_stats(*args)

    def evaluate(self, sample):
        """Given a graph (sample),
        computes the energy function of the Model

        Arguments:
            sample {GraphWrapper} -- input sample

        Returns:
            float -- resulting energy
        """
        interactions_count = sample.get_interactions_count(3)
        return np.dot(self._params, interactions_count)

    cdef get_local_energy(self, sample, edge, neigh):
        """Compute the local energy.

        Arguments:
            sample {GraphWrapper} -- Input sample
            edge {EdgeId} -- Edge identifier

        Returns:
            float -- Delta energy
        """
        cdef int edge_type = sample.get_edge_type(edge)
        cdef ndarray[double] interactions_count = np.zeros(3)

        for t in neigh:
            n_label = sample.get_edge_type(t)
            if n_label != edge_type:
                idx = edge_type + n_label - 1
                interactions_count[idx] += 1
        res = np.dot(self._params, interactions_count)

        return res

    @staticmethod
    def summary(results):
        data = dict()

        res = np.array([g.get_interactions_count(3) for g in results])
        data['beta01'] = res[:, 0]
        data['beta02'] = res[:, 1]
        data['beta12'] = res[:, 2]

        return data

    @staticmethod
    def sufficient_stats(result):
        return result.get_interactions_count(3)
