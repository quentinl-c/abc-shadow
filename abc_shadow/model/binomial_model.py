from math import log, exp


class BinomialModel(object):
    def __init__(self, n, theta):
        self.__n = n
        self.__theta = theta

    @property
    def n(self):
        return self.__n

    @property
    def theta(self):
        return self.__theta

    @theta.setter
    def theta(self, theta_new):
        self.__theta = theta_new

    def set_params(self, *args):

        if len(args) < 1:
            raise ValueError("⛔️ Given args lenght:{}, expected: 1".format(
                len(args)))

        self.theta = args[0]

    def evaluate(self, sample):
        res = sample * self.__theta
        res -= self.__n * log(1 + exp(self.__theta))
        return res

    def evaluate_from_stats(self, *args):
        if len(args) < 1:
            raise ValueError("⛔️ Given stats lenght:{}, expected: 1".format(
                len(args)))
        return self.evaluate(args[0])

    @staticmethod
    def summary(results):

        dataset = dict()
        dataset["successes"] = results

        return dataset
