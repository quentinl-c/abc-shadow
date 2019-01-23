import numpy as np
from math import exp, factorial

from .abc_impl import get_candidate


def mh_post_sampler(theta_0, y, delta, n, iters, ratio_fctn, mask=None):
    posteriors = list()

    theta_res = theta_0
    posteriors.append(theta_res)

    for i in range(iters - 1):
        if i % 1000 == 0:
            msg = "🔎  Iteration {} / {} n : {} theta {}".format(
                i, iters, n, theta_res)
            print(msg)

        theta_res = mh_algo(theta_res, y, delta, n, ratio_fctn, mask=mask)
        posteriors.append(theta_res)

    return posteriors


def mh_algo(theta_0, y, delta, n, ratio_fctn, mask=None):
    theta_res = np.array(theta_0)

    for _ in range(n):
        candidate = get_candidate(theta_res, delta, mask)
        density_ratio = ratio_fctn(y, theta_res, candidate)
        prob = np.random.uniform(0, 1)

        if prob < density_ratio:
            theta_res = candidate

    return theta_res


"""
Ratio Fonction (depending on which kind of model we want to sample)
"""


def norm_ratio(y, theta, candidate):
    m = len(y)

    q = (theta[1] / candidate[1])**(m/2)
    p1 = sum([(y_i - candidate[0])**2 for y_i in y]) / candidate[1]
    p2 = sum([(y_i - theta[0])**2 for y_i in y]) / theta[1]
    
    ratio = q * exp(-0.5*(p1 - p2))
    return ratio

def _binom_likelihood(n, p, k):
    res1 = factorial(int(n)) / (factorial(int(k)) * factorial(int(n - k)))
    res2 = p**(k) * (1 - p)**(n - k)
    res = res1 * res2

    return res

def binom_ratio(y, theta, candidate):
    n = theta[0]
    p_theta = theta[1]
    p_candidate = candidate[1]
    k = y[0]

    q1 = _binom_likelihood(n, p_candidate, k)
    q2 = _binom_likelihood(n, p_theta, k)
    ratio = q1 / q2

    return ratio

