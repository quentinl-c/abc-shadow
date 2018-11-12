import copy
import random
from math import exp

from .model.graph_model import GraphModel

DEFAULT_ITER = 100


def mh_sampler(sample, model, iters=DEFAULT_ITER):
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
    for _ in range(iters):
        results.append(copy.deepcopy(sample))

        # print("Iteration {}".format(it))
        for e in sample.get_elements():

            # Swap between old value and the new one (randomly taken)
            old_val = sample.get_particle(e)
            new_val = model.get_random_candidate_val()
            # new_val = 1

            delta = model.compute_delta(sample, e, new_val)

            epsilon = random.uniform(0, 1)

            if epsilon >= exp(delta):
                #print('rejected {}'.format(new_val))
                # Old value recovery
                sample.set_particle(e, old_val)
            #else:
                #print('Accepted {} -> {}'.format(old_val, new_val))

    return results
