from abc_shadow.model.norm_model import NormModel
from abc_shadow.abc_impl import abc_shadow, normal_sampler
import json
import numpy as np
import math


def main():
    """Experience sumary:
    * Random seed -> to make this experience more reproducible
    * Normal model may be expressed as follows (relatively constant):
    exp[theta1/theta2 * t1(y) - 0.5 * t2(y)/theta2]
       * theta1 is the mean parameter
       * theta2 is the variance parameter (std^2)
       * t1 is the sum of all sampled number sum(y_i)
       * t2 is the sum of all squared sampled number sum(y_i^2)
    * theta_prior is the vector of parameters estimated a priori
      (may be far from the real theta value)
    * y_obs corresponds to the observed sufficient statistics in our case: 
    [t1(y),t2(y)]
    * size -> corresponds to the sample's length
    """

    seed = 2018
    mean = 2
    var = 9
    theta_perfect = np.array([mean, var])

    theta_prior = np.array([10, 10])
    model = NormModel(theta_prior[0], theta_prior[1])
    size = 1000

    y_obs = normal_sampler(model, size, 1, seed=seed)

    # ABC Shadow parameters

    # Number of iterations in the shadow chain
    n = 200

    # Number of generated samples
    iters = 1000

    # Delta -> Bounds of proposal volume
    delta = np.array([0.05, 0.05])

    model.set_params(*theta_prior)
    posteriors = abc_shadow(model,
                            theta_prior,
                            y_obs,
                            delta,
                            n,
                            size,
                            iters,
                            sampler=normal_sampler)

    json_list = [post.tolist() if isinstance(post, np.ndarray)
                 else post for post in posteriors]

    with open('norm.json', 'w') as output_file:
        output_file.truncate()
        json.dump(json_list, output_file)

    print("🎉 🎉 🎉 END 🎉 🎉 🎉!")


if __name__ == '__main__':
    main()
