from abc_shadow.mh_impl import mh_post_sampler, norm_ratio

from math import sqrt
import json
import numpy as np


def main():
    """Experience sumary:
    * Random seed -> to make this experience more reproducible
    * Normal model may be expressed as a function of mean and a var
    * theta_0 is the vector of parameters estimated a priori
      (may be far from the real theta value)
    * y_obs corresponds to a list of random numbers
      generated according a Normal distribution
    * size -> corresponds to the sample's length
    """
    seed = 2018
    mean = 2
    var = 9

    theta_0 = np.array([10, 17])
    size = 100

    np.random.seed(seed)
    y_obs = np.random.normal(mean, sqrt(var), size)
    # Metropolis Hasting parameters
    # Number of iterations in the shadow chain
    n = 100

    # Number of generated samples
    iters = 100000

    # Delta -> Bounds of proposal volume
    delta = np.array([0.05, 0.05])

    posteriors = mh_post_sampler(theta_0, y_obs, delta, n, iters, norm_ratio)

    json_list = [post.tolist() if isinstance(post, np.ndarray)
                 else post for post in posteriors]

    with open('norm_mh_100000.json', 'w') as output_file:
        output_file.truncate()
        json.dump(json_list, output_file)

    print("🎉 🎉 🎉 END 🎉 🎉 🎉!")


if __name__ == '__main__':
    main()
