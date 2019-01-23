import json
import numpy as np

from abc_shadow.abc_impl import abc_shadow, metropolis_sampler
from abc_shadow.model.two_interactions_graph_model import \
    TwoInteractionsGraphModel
import time

def main():
    seed = 2018
    l1_param = 0.1
    l2_param = 0.1
    l12_param = 0.1

    np.random.seed(seed)
    sampler = metropolis_sampler

    theta = np.array([l1_param, l2_param, l12_param])
    model = TwoInteractionsGraphModel(*theta)

    size = 10
    print("=== Observation is being simulated ===")
    y_obs = sampler(model, size, 1000)

    # ABC Shadow parameters

    # Number of iterations in the shadow chain
    n = 100

    # Number of generated samples
    iters = 10

    # Delta -> Bounds of proposal volume
    delta = np.array([0.05, 0.05, 0.05])

    # Theta 0 far from exact parameters
    theta_0 = np.array([-1, 1, 1])

    print("====== ABC SHADOW ======")

    model.set_params(*theta_0)
    start = time.time()
    posteriors = abc_shadow(model,
                            theta_0,
                            y_obs,
                            delta,
                            n,
                            size,
                            iters,
                            sampler=sampler,
                            sampler_it=1000)
    print(time.time() - start)
    json_list = [post.tolist() if isinstance(post, np.ndarray)
                 else post for post in posteriors]

    with open('interactions_1000.json', 'w') as output_file:
        output_file.truncate()
        json.dump(json_list, output_file)

    print("🎉 🎉 🎉 END 🎉 🎉 🎉!")


if __name__ == '__main__':
    main()
