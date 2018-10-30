import json

import numpy as np

from abc_shadow.abc_impl import abc_shadow, bernouilli_sampler
from abc_shadow.model.bernouilli_model import BernouilliModel


def main():
    """Experience sumary:
    * Random seed -> to make this experience more reproducible
    * Binomial model may be expressed as an exponential model
    (relatively constant):
    exp[k * theta - n log(1+ exp(theta))]
       * theta is the single parameter
       * k is the number of successes happened in n trials
       * n is the number of trials (fixed)
    * theta_prior is the parameter estimated a priori
      (may be far from the real theta value)
    * y_obs corresponds to the observed sufficient statistics in our case: k
    * size -> corresponds to the sample's length
    """

    seed = 2018
    none_edge_param = 1
    edge_param = 1

    theta_perfect = np.array([none_edge_param, edge_param])

    theta_prior = np.array([none_edge_param, 10])
    model = BernouilliModel(theta_perfect[0], theta_perfect[1])
    size = 10

    y_obs = bernouilli_sampler(model, size, 100, seed)

    # ABC Shadow parameters

    # Number of iterations in the shadow chain
    n = 100

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
                            sampler=bernouilli_sampler,
                            sampler_it=100,
                            mask=[1, 0])  # n_edge_p is fixed

    json_list = [post.tolist() if isinstance(post, np.ndarray)
                 else post for post in posteriors]

    with open('bernouilli.json', 'w') as output_file:
        output_file.truncate()
        json.dump(json_list, output_file)

    print("🎉 🎉 🎉 END 🎉 🎉 🎉!")

if __name__ == '__main__':
    main()
