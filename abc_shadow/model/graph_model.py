from abc import ABC, abstractmethod
import numpy.random as random


class GraphModel(ABC):

    type_values = {0, 1}

    @abstractmethod
    def get_local_energy(sample, edge, neigh=None):
        pass

    @staticmethod
    @abstractmethod
    def summary(results):
        pass

    @abstractmethod
    def evaluate_from_stats(self, *args):
        pass

    @abstractmethod
    def evaluate(self, sample):
        pass

    def compute_delta(self, mut_sample, edge, new_val):
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
        neigh = [mut_sample.get_edge_type(n) for n in mut_sample.graph.neighbors(edge)]
        old_energy = self.get_local_energy(mut_sample, edge, neigh=neigh)

        mut_sample.set_edge_type(edge, new_val)

        # Computes the delta between old and new energy
        new_energy = self.get_local_energy(mut_sample, edge, neigh=neigh)
        delta = new_energy - old_energy

        return delta

    @classmethod
    def get_random_candidate_val(cls, p=None):
        return random.choice(list(cls.type_values), p=p)
