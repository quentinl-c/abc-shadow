import numpy as np
from .model import Model


class NormModel(Model):

    min_extrem = -10
    max_extrem = 10

    def __init__(self, *args):
        if len(args) != 2:
            raise ValueError
        super().__init__(*args)

    @property
    def mean(self):
        return self._params[0]

    @property
    def var(self):
        return self._params[1]

    def set_params(self, *args):

        if len(args) != 2:
            raise ValueError
        super().__init__(*args)

    def evaluate(self, sample):
        exp = (self.mean / self.var) * sample.get_sample_sum()
        exp -= (2 * self.var)**(-1) * sample.get_sample_square_sum()
        return exp

    def evaluate_from_stats(self, *args):

        if len(args) != 2:
            raise ValueError("⛔️ Given stats lenght:{}, expected: 2".format(
                len(args)))

        res = (self.mean / self.var) * args[0]
        res -= (1/(2 * self.var)) * args[1]
        return res

    @staticmethod
    def summary_dict(results):

        dataset = dict()
        dataset["sum"] = [sum(r) for r in results]
        dataset["square_sum"] = [sum(np.array(r)**2) for r in results]

        return dataset
