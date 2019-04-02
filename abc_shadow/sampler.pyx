from libc.math cimport exp
import numpy as np
cimport numpy as np
from numpy cimport ndarray
import cython
from .model.graph_model import GraphModel

DEFAULT_ITER = 100


cpdef mcmc_sampler(sample, model, iters=DEFAULT_ITER, burnin=1, by=1):
    return _mcmc_sampler(sample, model, iters=iters, burnin=burnin, by=by)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef list _mcmc_sampler(sample, model, iters=DEFAULT_ITER, burnin=1, by=1):
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
    cdef list results = list()
    cdef int old_val, new_val, i
    cdef double epsilon
    cdef np.float_t delta
    cdef ndarray[double] params = np.array(model.params)
    cdef ndarray[double] stats = np.array(model.get_stats(sample))
    cdef ndarray[double] delta_stats = np.zeros(len(params))

    # for i in range(burnin + by * iters):
    for i in range(iters):
        if i >= burnin and i % by == 0:
            results.append(stats.copy())

        for e in sample.get_elements():

            old_val = sample.get_edge_type(e)
            # Naw random val is choosed
            new_val = model.get_random_candidate_val()
            delta_stats = model.get_delta_stats(sample, e, new_val)
            delta = np.dot(params, delta_stats)
            #delta = model.compute_delta(sample, e, new_val)
            epsilon = np.random.uniform(0, 1)
            if epsilon >= exp(delta):
                # Rejected
                sample.set_edge_type(e, old_val)
            else:
                stats += delta_stats

    return results
