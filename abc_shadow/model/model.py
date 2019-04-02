import numpy as np
from abc import ABC, abstractmethod


class Model(ABC):

    def __init__(self, *params):
        self._params = list(params)

    def set_params(self, *args):
        self._params = list(args)

    def evaluate_from_stats(self, *args):
        return np.dot(self._params, args)

    @property
    def params(self):
        return self._params

    @staticmethod
    @abstractmethod
    def summary(results):
        pass

    @abstractmethod
    def evaluate(self, sample):
        pass
