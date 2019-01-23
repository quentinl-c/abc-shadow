import copy
from math import exp
import numpy as np
from .model.graph_model import GraphModel

DEFAULT_ITER = 100


def mcmc_sampler(sample, model, iters=DEFAULT_ITER, burnin=1, by=1):
    """Executes Metropolis Hasting sampler algorith

    Arguments:
        sample {GraphWrapper} -- Initial graph on which MH is applied
        model {Model} -- Model difining the energy function

    Keyword Arguments:
        iter {int} -- Number of iterations (default: {DEFAULT_ITER})

    Raises:
        ValueError -- Only graphs can be sampled
                      (check if the give model is a GraphModel)

    Returns:
        list[GraphWrapper] -- List of resulting graphs for each iteration
    """

    if not isinstance(model, GraphModel):
        err_msg = "⛔️ mh_sampler function can only sample graph"
        raise ValueError(err_msg)

    # resulting list
    results = list()
    # rejected = 0
    # accepted = 0
    for i in range(burnin + by * iters):
        if i >= burnin and i % by == 0:
            results.append(sample.copy())

        # print("Iteration {}".format(it))
        for e in sample.get_elements():

            # Swap between old value and the new one (randomly taken)
            old_val = sample.get_edge_type(e)
            new_val = model.get_random_candidate_val()
            # new_val = 1

            delta = model.compute_delta(sample, e, new_val)

            epsilon = np.random.uniform(0, 1)

            if epsilon >= exp(delta):
                # rejected += 1
                # print('rejected {}'.format(new_val))
                # Old value recovery
                sample.set_particle(e, old_val)
    #         else:
    #             accepted += 1
    #             # print('Accepted {} -> {}'.format(old_val, new_val))
    # print("Number of rejected proposals: {}".format(rejected))
    # print("Number of accepted proposals: {}".format(accepted))
    # print("ratio accepted / rejected: {}".format(accepted / rejected))

    return results
