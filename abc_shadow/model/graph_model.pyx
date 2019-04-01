# distutils: language = c++
# cython: boundscheck = False
from model cimport Model
import numpy.random as random


cdef class GraphModel(Model):

    type_values = {0, 1}

    cdef double _compute_delta(self, mut_sample, cpp_edge edge, int new_val):
        """Given a graph sample (mut_sample), an edge on which we will
        affect the new attribute value (new_val),
        computes difference between the new energy (on the modified sample)
        and the previous one (on the initial sample).
        Instead of counting all directe edges,
        computes only the difference between x_new - x_old

        Arguments:
            mut_sample {GraphWrapper} -- initial sample
                                         (mutable - reference passing)
                                         by side effect, one will be modified
            edge {tuple(int,int)} -- designated edge
                                     (for which the attribute will be modified)
            new_val {int} -- new attribute value comprise

        Returns:
            float -- Energy delta between modified sample and initial one
        """
        # neigh = [mut_sample.vertex[n] for n in mut_sample.get_edge_neighbourhood(edge)]
        cdef cpp_vector[cpp_edge] neigh = mut_sample.get_edge_neighbourhood(edge)
        cdef double old_energy = self.get_local_energy(mut_sample, edge, neigh=neigh)

        mut_sample.set_edge_type(edge, new_val)

        # Computes the delta between old and new energy
        cdef double new_energy = self.get_local_energy(mut_sample, edge, neigh=neigh)
        cdef double delta = new_energy - old_energy

        return delta

    cpdef double compute_delta(self, mut_sample, cpp_edge edge, int new_val):
        return self._compute_delta(mut_sample, edge, new_val)

    @classmethod
    def get_random_candidate_val(cls, p=None):
        return random.choice(list(cls.type_values), p=p)
