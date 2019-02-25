from math import log, exp
from .model import Model


class BinomialModel(Model):
    def __init__(self, *args):
        if len(args) != 2:
            raise ValueError

        super().__init__(*args)

    @property
    def n(self):
        return self._params[0]

    @property
    def theta(self):
        return self._params[1]

    @theta.setter
    def theta(self, new_theta):
        self._params[1] = new_theta

    def set_params(self, *args):

        if len(args) != 1:
            raise ValueError("⛔️ Given args lenght:{}, expected: 1".format(
                len(args)))

        self.theta = args[0]

    def evaluate(self, sample):
        res = sample * self.__theta
        res -= self.__n * log(1 + exp(self.__theta))
        return res

    def evaluate_from_stats(self, *args):
        if len(args) != 1:
            raise ValueError("⛔️ Given stats lenght:{}, expected: 1".format(
                len(args)))
        return self.evaluate(args[0])

    @staticmethod
    def summary(results):

        dataset = dict()
        dataset["successes"] = results

        return dataset
