
from abc_shadow.mh_impl import mh_post_sampler, binom_ratio

import json
import numpy as np
from math import exp

def main():
    """Experience sumary:
    * Random seed -> to make this experience more reproducible
    * Binomial model 
        n!/k!(n-k)! * p^k (1 - p)^(n-k)
       * p is the probability of success
       * k is the number of successes happened in n trials
       * n is the number of trials (fixed)
    * p_0 is the probability estimated a priori
      (may be far from the real theta value)
    * y_obs corresponds to the observed sufficient statistic in our case: k
    * size -> corresponds to the sample's length = 1 (in our case)
    """

    seed = 2018
    theta = 0
    theta_0 = 10
    p = exp(theta) / (1 + exp(theta))
    p_0 = exp(theta_0) / (1 + exp(theta_0))
    n_p = 100 # fixed
    theta_0 = np.array([n_p, p_0])
    print(theta_0)
    print(p)
    size = 1

    np.random.seed(seed)
    y_obs = np.random.binomial(n_p, p, size)

    # ABC Shadow parameters

    # Number of iterations in the shadow chain
    n = 100

    # Number of generated samples
    iters = 100000

    # Delta -> Bounds of proposal volume
    delta = np.array([0.005, 0.005])

    posteriors = mh_post_sampler(theta_0, y_obs, delta, n, iters, binom_ratio, mask=[1,0])


    json_list = [post.tolist() if isinstance(post, np.ndarray)
                 else post for post in posteriors]

    with open('binom_mh_100000.json', 'w') as output_file:
        output_file.truncate()
        json.dump(json_list, output_file)

    print("🎉 🎉 🎉 END 🎉 🎉 🎉!")


if __name__ == '__main__':
    main()
