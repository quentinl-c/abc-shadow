import random
from json import JSONEncoder
from math import exp, sqrt

import numpy as np
import numpy.ma as ma
from .model.graph_model import GraphModel
from .graph.graph_wrapper import GraphWrapper
from .sampler import mh_sampler
import copy

"""
Implementation of ABC Shadow Algorithm
"""


def abc_shadow(model, theta_0, y, delta, n, size, iters,
               sampler=None, sampler_it=1, mask=None):
    """Executes ABC posterior sampling
    
    Arguments:
        model {GraphModel | Model} -- Model encompassing phenomenon of nterest
        theta_0 {[type]} -- Initial parameter (must match to model's parameters)
        y {List[Sample]} -- Obseved sample(s)
        delta {List[float]} -- Bounds of proposal volumes for each parameter
        n {in} -- number of iteration for the shadow chain
        size {int} -- Size of sample(s) sampled at initialisation of the shadow chain
        iters {int} -- Number of posterior samples
    
    Keyword Arguments:
        sampler {fct} -- Sampler function respeciting the following constraint on arguments 
                         args: model, size, it, seed=None
                         (default: {metropolis_hasting graph sampler})
        sampler_it {int} -- Number of sampler iteration  (default: {1})
        mask {[type]} -- Mask array to fix some parameter (by setting 1 at the parameter position)
                         (default: {None})
    
    Returns:
        List[np.array] -- List of sampled parameters -> posterior distribution
    """


    posteriors = list()

    theta_res = theta_0
    posteriors.append(theta_res)

    for i in range(iters):
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

        if i % 10 == 0:
            msg = "🔎  Iteration {} / {} n : {}, samplerit {}, theta {}".format(
                i, iters, n, sampler_it, theta_res)
            print(msg)

    return posteriors


def abc_shadow_chain(model, theta_0, y, delta, n, size, sampler_it,
                     sampler=None, mask=None):
    """Executes ABC Shdow chain algorithm
    
    Arguments:
        model {GraphModel | Model} -- Model encompassing phenomenon of nterest
        theta_0 {[type]} -- Initial parameter (must match to model's parameters)
        y {List[Sample]} -- Obseved sample(s)
        delta {List[float]} -- Bounds of proposal volumes for each parameter
        n {in} -- number of iteration for the shadow chain
        size {int} -- Size of sample(s) sampled at initialisation of the shadow chain
        sampler_it {int} -- Number of sampler iteration
  
    Keyword Arguments:
       sampler {fct} -- Sampler function respeciting the following constraint on arguments 
                         args: model, size, it, seed=None
                         (default: {metropolis_hasting graph sampler})
       mask {[type]} -- Mask array to fix some parameter (by setting 1 at the parameter position)
                         (default: {None})
    
    Returns:
        np.array -- Last accepted candidate parameter
    """


    model.set_params(*theta_0)
    theta_res = np.array(theta_0)

    if sampler is not None:
        y_sim = sampler(model, size, sampler_it)

    else:
        y_sim = metropolis_sampler(model, size, sampler_it)
    for _ in range(n):
        candidate = get_candidate(theta_res, delta, mask)
        alpha = get_shadow_density_ratio(model, y, y_sim, theta_res, candidate)

        prob = random.uniform(0, 1)

        if alpha > prob:
            theta_res = candidate
    return theta_res


def get_candidate(theta, delta, mask=None):
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

        indices = set(range(len(theta))) - set(np.nonzero(mask)[0])
    else:
        indices = range(len(theta))

    if not indices:
        return candidate_vector

    candidate_indice = random.sample(indices, 1)[0]

    d = delta[candidate_indice]
    old = candidate_vector[candidate_indice]
    candidate_value = np.random.uniform(old - d / 2, old + d/2)

    candidate_vector[candidate_indice] = candidate_value

    return candidate_vector


def get_shadow_density_ratio(model, y_obs, y_sim, theta, candidate):
    model.set_params(*candidate)
    p1 = model.evaluate_from_stats(*y_obs)
    q2 = model.evaluate_from_stats(*y_sim)

    model.set_params(*theta)
    p2 = model.evaluate_from_stats(*y_obs)
    q1 = model.evaluate_from_stats(*y_sim)

    ratio = (exp(p1 - p2)) * (exp(q1 - q2))
    alpha = min(1, ratio)

    return alpha

"""
===============
Samplers
> Functions used to generate samples from a given probability density function
===============
"""


def normal_sampler(model, size, it, seed=None):
    samples = list()

    for _ in range(it):
        if seed is not None:
            np.random.seed(seed)

        sample = np.random.normal(model.get_mean(),
                                  sqrt(model.get_var()), size)
        samples.append(sample)

    y_sim = [np.average(stats) for stats in model.summary(samples).values()]
    return np.array(y_sim)


def binom_sampler(model, size, it, seed=None):
    samples = list()
    theta = model.get_theta()
    p = exp(theta) / (1 + exp(theta))
    n = model.get_n()

    for _ in range(it):
        if seed is not None:
            np.random.seed(seed)

        sample = np.random.binomial(n, p, size)
        samples.append(sample)

    y_sim = [np.average(stats) for stats in model.summary(samples).values()]
    return np.array(y_sim)

"""
Graph Sampler
"""


def metropolis_sampler(model, size, mh_sampler_it):
    
    if isinstance(model, GraphModel):
        init_sample = GraphWrapper(size)
    else:
        err_msg = "⛔️ metropolis_sampler wrapper may only be used" \
                  "to sample graph"
        raise ValueError(err_msg)

    samples = mh_sampler(init_sample, model, mh_sampler_it)

    summary = model.summary(samples)

    vec = [np.average(stats) for stats in summary.values()]
    return np.array(vec)


def binom_graph_sampler(model, size, it, seed=None):
    sample = GraphWrapper(size)

    seeds = None
    if seed is not None:
        np.random.seed(seed)
        size = it * len(sample.get_elements())
        seeds = iter(np.random.random_integers(0, 1000, size=size))

    none_edge_param = model.get_none_edge_param()
    edge_param = model.get_edge_param()

    none_edge_prob = none_edge_param / (edge_param + none_edge_param)
    edge_prob = edge_param / (edge_param + none_edge_param)
    probs = [none_edge_prob, edge_prob]

    dist = list()

    for i in range(it):
        for edge in sample.get_elements():

            if seeds is not None:
                s = seeds.__next__()
                np.random.seed(s)

            edge_attr = np.random.choice(a=list(model.type_values),
                                         size=1,
                                         replace=False,
                                         p=probs)
            sample.set_edge_type(edge, edge_attr[0])

        dist.append(copy.deepcopy(sample))

    mean_stats = (np.mean([s.get_none_edge_count() for s in dist]),
                  np.mean([s.get_edge_count() for s in dist]))
    return mean_stats
