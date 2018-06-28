import numpy as np


class NormModel(object):

    min_extrem = -10
    max_extrem = 10

    def __init__(self, mean, var):
        self.__mean = mean
        self.__var = var  # varaiance

    def get_mean(self):
        return self.__mean

    def get_var(self):
        return self.__var

    def set_mean_param(self, new_mean):
        self.__mean = new_mean

    def set_var_param(self, new_var):
        self.__var = new_var

    def set_params(self, *args):

        if len(args) < 2:
            raise ValueError

        self.set_mean_param(args[0])
        self.set_var_param(args[1])

    def evaluate(self, sample):
        exp = (self.__mean / self.__var) * sample.get_sample_sum()
        exp -= (2 * self.__var)**(-1) * sample.get_sample_square_sum()
        return exp

    def evaluate_from_stats(self, stats):

        if len(stats) < 2:
            raise ValueError("Given args lenght:{}, expected: 2".format(
                len(args)))
        expr = (self.__mean / self.__var) * stats[0]
        expr -= (1/(2 * self.__var)) * stats[1]
        return expr

    def compute_delta(self, sample, id, new_val):
        old_energy = self.evaluate(sample)

        sample.set_particle(id, new_val)
        new_energy = self.evaluate(sample)

        delta = old_energy - new_energy
        return delta

    @classmethod
    def get_random_candidate_val(cls):
        return np.random.uniform(cls.min_extrem, cls.max_extrem)

    @staticmethod
    def summary(results):

        dataset = dict()
        dataset["sum"] = [sum(r) for r in results]
        dataset["square_sum"] = [sum(np.array(r)**2) for r in results]

        return dataset
