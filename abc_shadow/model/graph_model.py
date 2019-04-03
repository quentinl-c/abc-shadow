from .model import Model
import numpy.random as random
from abc import abstractmethod


class GraphModel(Model):

    type_values = {0, 1}

    @classmethod
    def get_random_candidate_val(cls, p=None):
        return random.choice(list(cls.type_values), p=p)

    @abstractmethod
    def get_local_energy(self, sample, edge, neigh):
        pass

    @staticmethod
    @abstractmethod
    def get_delta_stats(mut_sample, edge, new_label):
        pass
