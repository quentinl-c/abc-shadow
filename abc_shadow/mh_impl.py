import numpy as np
import random
from math import exp, pi

from .abc_impl import get_candidate


def mh_post_sampler(theta_0, y, delta, n, iters, ratio_fctn):
    posteriors = list()

    theta_res = theta_0
    posteriors.append(theta_res)

    for i in range(iters):
        theta_res = mh_algo(theta_res, y, delta, n, ratio_fctn)
        posteriors.append(theta_res)
        if i % 10 == 0:
            msg = "🔎  Iteration {} / {} n : {} theta {}".format(
                i, iters, n, theta_res)
            print(msg)

    return posteriors


def mh_algo(theta_0, y, delta, n, ratio_fctn):
    theta_res = np.array(theta_0)

    for _ in range(n):
        candidate = get_candidate(theta_res, delta)
        density_ratio = ratio_fctn(y, theta_res, candidate)
        prob = random.uniform(0, 1)

        if prob < density_ratio:
            theta_res = candidate

    return theta_res


"""
Ration Fonction (depending on which kind of model we want to sample)
"""


def norm_ratio(y, theta, candidate):
    m = len(y)

    q = (theta[1] / candidate[1])**(m/2)
    p1 = sum([(y_i - candidate[0])**2 for y_i in y]) / candidate[1]
    p2 = sum([(y_i - theta[0])**2 for y_i in y]) / theta[1]
    
    ratio = q * exp(-0.5*(p1 - p2))
    return ratio


def binom_ratio():
    pass
