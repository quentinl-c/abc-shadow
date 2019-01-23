import json

import numpy as np
from abc_shadow.model.markov_star_graph_model import MarkovStarGraphModel
from abc_shadow.graph.graph_wrapper import GraphWrapper
from abc_shadow.abc_impl import (abc_shadow,
                                 metropolis_sampler)
import networkx as nx


def main():
    seed = 2018
    edge_param = -1.656959
    two_star_param = 0.008802

    np.random.seed(seed)

    sampler = metropolis_sampler

    theta_0 = np.array([edge_param, two_star_param])

    input_data = get_flomarriage_graph()

    size = input_data.get_initial_dim()

    obs_edges = input_data.get_edge_count()
    obs_two_stars = input_data.get_two_star_count()

    y_obs = np.array([obs_edges, obs_two_stars])

    model = MarkovStarGraphModel(*theta_0)

    # y_obs = sampler(model, size, 100)
    print(y_obs)
    # ABC Shadow parameters
    return
    # Number of iterations in the shadow chain
    n = 50

    # Number of generated samples
    iters = 10000

    # Delta -> Bounds of proposal volume
    delta = np.array([0.005, 0.005])

    model.set_params(*theta_0)
    posteriors = abc_shadow(model,
                            theta_0,
                            y_obs,
                            delta,
                            n,
                            size,
                            iters,
                            sampler=sampler,
                            sampler_it=100)

    json_list = [post.tolist() if isinstance(post, np.ndarray)
                 else post for post in posteriors]

    with open('markov_flomarriage_100_100000.json', 'w') as output_file:
        output_file.truncate()
        json.dump(json_list, output_file)

    print("🎉 🎉 🎉 END 🎉 🎉 🎉!")


def get_flomarriage_graph():
    g = nx.florentine_families_graph()
    g.add_node('Pucci')

    g_wr = GraphWrapper(gr=g)
    return g_wr

if __name__ == "__main__":
    main()
