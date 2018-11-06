import json

import numpy as np

from abc_shadow.abc_impl import abc_shadow, binomial_grapb_sampler
from abc_shadow.model.binomial_graph_model import BinomialGraphModel


def main():
    """Experience sumary:
    * Random seed -> to make this experience more reproducible
    * Bernouilli model may be expressed as an exponential model
    (relatively constant):
    exp[theta1 * t1(y) + theta2 * t2(y))]
       * theta1 is the none_edge parameter
         representing the portion of none edge in a graph (fixed)
       * theta2 is the edge parameter
         representing the portion of existing edge in a graph
       * t1(y) is the number of none edges in the observed graph
         (not existing edges)
       * t2(y) is the number of edges in the observed graph (existing)
    * theta_prior is the parameter estimated a priori
      (may be far from the real theta value)
    * y_obs corresponds to the observed sufficient statistics in our case: 
      t1(y_obs), t2(y_obs)
    * size -> corresponds to the graph dimension ie. number of nodes
    """

    seed = 2018
    none_edge_param = 1
    edge_param = 1

    theta_perfect = np.array([none_edge_param, edge_param])

    theta_prior = np.array([none_edge_param, 10])
    model = BinomialGraphModel(theta_perfect[0], theta_perfect[1])
    size = 10

    y_obs = binomial_grapb_sampler(model, size, 100, seed)

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
                            sampler=binomial_grapb_sampler,
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
