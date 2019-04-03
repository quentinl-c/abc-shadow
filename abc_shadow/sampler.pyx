from libc.math cimport exp
import numpy as np
cimport numpy as np
from numpy cimport ndarray
import cython
from .model.graph_model import GraphModel

DEFAULT_ITER = 100


cpdef mcmc_sampler(sample, model, iters=DEFAULT_ITER, burnin=1, by=1):
    """wrapper of cythonized function _mcmc_sampler
    
    Arguments:
        sample {GraphWrapper} -- Initial graph on which MH is applied
        model {GraphModel} -- Model difining the energy function
    
    Keyword Arguments:
        iter {int} -- Number of iterations (default: {DEFAULT_ITER})
        burnin {int} -- burn in iterations (default: {1})
        by {int} -- sampling ratio (default: {1})
    
    Returns:
        list[ndarray[int]] -- Sufficient statistics list of resulting sampled graphs
    """
    return _mcmc_sampler(sample, model, iters, burnin, by)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef list _mcmc_sampler(object sample, object model, int iters, int burnin, int by):
    """Executes Metropolis Hastings sampler algorith

    Arguments:
        sample {GraphWrapper} -- Initial graph on which MH is applied
        model {GraphModel} -- Model difining the energy function
        iter {int} -- Number of iterations (default: {DEFAULT_ITER})
        burnin {int} -- burn in iterations (default: {1})
        by {int} -- sampling ratio (default: {1})

    Raises:
        ValueError -- Only graphs can be sampled
                      (check if the give model is a GraphModel)

    Returns:
        list[ndarray[int]] -- Sufficient statistics list of resulting sampled graphs
    """

    if not isinstance(model, GraphModel):
        err_msg = "⛔️ mh_sampler function can only sample graph"
        raise ValueError(err_msg)

    #  ++++ resulting list of sufficient statistics ++++
    cdef list results = list()

    # ++++ Definition of variables ++++ 
    cdef int old_val, new_val, i
    cdef double epsilon
    cdef np.float_t delta

    # ++++ Model parameters ++++
    cdef ndarray[double] params = np.array(model.params)

    # ++++ Sufficient statistics of the initial graph ++++
    cdef ndarray[double] stats = np.array(model.get_stats(sample))

    # ++++ Initialisation of sufficient statistics delta vector ++++
    cdef ndarray[double] delta_stats = np.zeros(len(params))
    
    # ++++ potential values an edge may take ++++
    cdef list potential_values = model.type_values

    # ++++ All edges of the graph ++++
    cdef list edges = list(sample.get_elements())

    # ++++ new labels randomly generated ++++
    cdef ndarray[long] new_labels


    for i in range(iters):
        new_labels = np.random.choice(potential_values, size=len(edges))
        
        if i >= burnin and i % by == 0:
            results.append(stats.copy())

        for j in range(len(edges)):
            e = edges[j]
            old_val = sample.get_edge_type(e)
            new_val = new_labels[j]
            
            # Compute the delta
            delta_stats = model.get_delta_stats(sample, e, new_val)
            delta = np.dot(params, delta_stats)

            epsilon = np.random.uniform(0, 1)
            if epsilon >= exp(delta):
                # Rejected
                sample.set_edge_type(e, old_val)
            else:
                # Accepted
                stats += delta_stats

    return results
