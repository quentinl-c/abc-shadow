from abc_shadow.model.binomial_model import BinomialModel
from abc_shadow.abc_impl import abc_shadow, binom_sampler
import json
import numpy as np


def main():
    """Experience sumary:
    * Random seed -> to make this experience more reproducible
    * Binomial model may be expressed as an exponential model
    (relatively constant):
    exp[k * theta - n log(1+ exp(theta))]
       * theta is the single parameter
       * k is the number of successes happened in n trials
       * n is the number of trials (fixed)
    * theta_0 is the parameter estimated a priori
      (may be far from the real theta value)
    * y_obs corresponds to the observed sufficient statistics in our case: k
    * size -> corresponds to the sample's length
    """

    seed = 2018
    theta = 0
    n_p = 100  # fixed
    theta_perfect = np.array([theta])

    theta_0 = np.array([10])
    model = BinomialModel(n_p, theta_perfect[0])
    size = 1

    np.random.seed(seed)
    y_obs = binom_sampler(model, size, 1)

    # ABC Shadow parameters

    # Number of iterations in the shadow chain
    n = 100

    # Number of generated samples
    iters = 100000

    # Delta -> Bounds of proposal volume
    delta = np.array([0.05])

    model.set_params(*theta_0)
    posteriors = abc_shadow(model,
                            theta_0,
                            y_obs,
                            delta,
                            n,
                            size,
                            iters,
                            sampler=binom_sampler)

    json_list = [post.tolist() if isinstance(post, np.ndarray)
                 else post for post in posteriors]

    with open('binom_abc_100000.json', 'w') as output_file:
        output_file.truncate()
        json.dump(json_list, output_file)

    print("🎉 🎉 🎉 END 🎉 🎉 🎉!")

if __name__ == '__main__':
    main()
