from libc.math cimport exp, sqrt

import numpy as np
from .model.graph_model import GraphModel
from .graph.graph_wrapper import GraphWrapper
cimport numpy as np
from numpy cimport ndarray,float64_t

from .sampler import mcmc_sampler
import copy

INF = -100
SUP = 100

"""
Implementation of ABC Shadow Algorithm
"""


cpdef abc_shadow(model, theta_0, y, delta, n, size, iters,
               sampler=None, sampler_it=1, mask=None):
    """Executes ABC posterior sampling

    Arguments:
        model {GraphModel | Model} -- Model encompassing phenomenon of nterest
        theta_0 {[type]} -- Initial parameter (must match to model's
                            parameters)
        y {List[Sample]} -- Obseved sample(s)
        delta {List[float]} -- Bounds of proposal volumes for each parameter
        n {in} -- number of iteration for the shadow chain
        size {int} -- Size of sample(s) sampled at initialisation
                       of the shadow chain
        iters {int} -- Number of posterior samples

    Keyword Arguments:
        sampler {fct} -- Sampler function respeciting the following constraint
                         on arguments
                         args: model, size, it, seed=None
                         (default: {metropolis_hasting graph sampler})
        sampler_it {int} -- Number of sampler iteration  (default: {1})
        mask {[type]} -- Mask array to fix some parameter (by setting 1 at 
                         the parameter position)
                         (default: {None})

    Returns:
        List[ndarray] -- List of sampled parameters -> posterior distribution
    """

    cdef list posteriors = list()

    cdef ndarray[double] theta_res = theta_0
    posteriors.append(theta_res)

    for i in range(iters - 1):

        if i % 100 == 0:
            msg = "🔎  Iteration {} / {} n : {}, samplerit {}, theta {}".format(
                i, iters, n, sampler_it, theta_res)
            print(msg)

        theta_res = abc_shadow_chain(model,
                                     theta_res,
                                     y,
                                     delta,
                                     n,
                                     size,
                                     sampler=sampler,
                                     sampler_it=sampler_it,
                                     mask=mask)

        posteriors.append(theta_res)

    return posteriors


cdef abc_shadow_chain(model, theta_0, y, delta, n, size, sampler_it,
                     sampler=None, mask=None):
    """Executes ABC Shdow chain algorithm

    Arguments:
        model {GraphModel | Model} -- Model encompassing phenomenon of nterest
        theta_0 {[type]} -- Initial parameter
                            (must match to model's parameters)
        y {List[Sample]} -- Obseved sample(s)
        delta {List[float]} -- Bounds of proposal volumes for each parameter
        n {in} -- number of iteration for the shadow chain
        size {int} -- Size of sample(s) sampled at initialisation
                       of the shadow chain
        sampler_it {int} -- Number of sampler iteration

    Keyword Arguments:
       sampler {fct} -- Sampler function respeciting the following constraint
                         on arguments
                         args: model, size, it, seed=None
                         (default: {metropolis_hasting graph sampler})
       mask {[type]} -- Mask array to fix some parameter
                         (by setting 1 at the parameter position)
                         (default: {None})

    Returns:
        np.array -- Last accepted candidate parameter
    """

    model.set_params(*theta_0)
    cdef ndarray[double] theta_res = np.array(theta_0)
    cdef ndarray[double] candidate

    if sampler is not None:
        y_sim = sampler(model, size, sampler_it)

    else:
        y_sim = metropolis_sampler(model, size, sampler_it)

    for _ in range(n):
        candidate = get_candidate(theta_res, delta, mask)
        alpha = get_shadow_density_ratio(model, y, y_sim, theta_res, candidate)

        prob = np.random.uniform(0, 1)

        if alpha > prob:
            theta_res = candidate
    return theta_res


cpdef get_candidate(theta, delta, mask=None):
    """Get a candidate vector theta prime
       picked from a Uniform distribution centred on theta (old)
       with a volume bound: delta

    Arguments:
        theta {array[float]} -- intial theta parameter
        delta {array[float]} -- volume : delta

    Keyword Arguments:
        mask {array[bool /int{0,1}]} -- maskek array to fix theta element
                                        (default: {None})
                        example:
                         get_candidate([1,2,3], [0.001, 0.002, 0.003], [1,1,0])
                         array([1., 2. , 3.00018494])
                         The first two elements keep theta values

    Returns:
        array[float] -- picked theta prime
    """

    if len(delta) != len(theta):
        err_msg = "⛔️ delta array should have the same length as theta"
        raise ValueError(err_msg)

    candidate_vector = np.array(theta, dtype=float, copy=True)

    if mask is not None:
        if len(mask) != len(theta):
            err_msg = "🤯 mask array should have the same length as theta"
            raise ValueError(err_msg)

        indices = list(set(range(len(theta))) - set(np.nonzero(mask)[0]))
    else:
        indices = range(len(theta))

    if not indices:
        return candidate_vector

    candidate_indice = np.random.choice(indices)

    d = delta[candidate_indice]
    old = candidate_vector[candidate_indice]
    candidate_value = np.random.uniform(old - d / 2, old + d/2)

    candidate_vector[candidate_indice] = candidate_value if INF < candidate_value < SUP else old

    return candidate_vector


cdef float get_shadow_density_ratio(model, y_obs, y_sim, theta, candidate):
    model.set_params(*candidate)
    cdef float64_t p1 = model.evaluate_from_stats(*y_obs)
    cdef float64_t q2 = model.evaluate_from_stats(*y_sim)
    model.set_params(*theta)
    cdef float64_t p2 = model.evaluate_from_stats(*y_obs)
    cdef float64_t q1 = model.evaluate_from_stats(*y_sim)

    cdef float ratio = (exp(p1 - p2)) * (exp(q1 - q2))
    cdef float alpha = min(1, ratio)

    return alpha


"""
===============
Samplers
> Functions used to generate samples from a given probability density function
===============
"""


cpdef normal_sampler(model, size, it):
    samples = list()

    for _ in range(it):

        sample = np.random.normal(model.mean,
                                  sqrt(model.var), size)
        samples.append(sample)

    y_sim = [np.average(stats) for stats in model.summary_dict(samples).values()]
    return np.array(y_sim)


cpdef binom_sampler(model, size, it):
    samples = list()
    theta = model.theta
    p = exp(theta) / (1 + exp(theta))
    n = model.n

    for _ in range(it):

        sample = np.random.binomial(n, p, size)
        samples.append(sample)

    y_sim = [np.average(stats) for stats in model.summary_dict(samples).values()]
    return np.array(y_sim)


"""
Graph Sampler
"""


cpdef metropolis_sampler(model, size, mh_sampler_it):

    if not isinstance(model, GraphModel):
        err_msg = "⛔️ metropolis_sampler wrapper may only be used" \
                  "to sample graph"
        raise ValueError(err_msg)

    cdef object init_sample = GraphWrapper(size)
    cdef list stat_samples = mcmc_sampler(init_sample, model, mh_sampler_it)

    cdef ndarray vec = np.mean(stat_samples, axis=0)
    return vec


cpdef binom_graph_sampler(model, size, it):
    sample = GraphWrapper(size)

    none_edge_param = model.none_edge_param
    edge_param = model.edge_param

    none_edge_prob = none_edge_param / (edge_param + none_edge_param)
    edge_prob = edge_param / (edge_param + none_edge_param)
    probs = [none_edge_prob, edge_prob]

    dist = list()

    for i in range(it):
        for edge in sample.get_elements():
            edge_attr = model.get_random_candidate_val(p=probs)

            sample.set_edge_type(edge, edge_attr)

        dist.append(copy.deepcopy(sample))

    mean_stats = (np.mean([s.get_none_edge_count() for s in dist]),
                  np.mean([s.get_edge_count() for s in dist]))
    return mean_stats
