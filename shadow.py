#!/usr/bin/env python3
import argparse
import configparser
import json
import os
import time
from math import exp, sqrt

import numpy as np

from abc_shadow.abc_impl import (abc_shadow, binom_sampler, metropolis_sampler,
                                 normal_sampler)
from abc_shadow.mh_impl import binom_ratio, mh_post_sampler, norm_ratio
from abc_shadow.model.binomial_edge_graph_model import BinomialEdgeGraphModel
from abc_shadow.model.binomial_graph_model import BinomialGraphModel
from abc_shadow.model.binomial_model import BinomialModel
from abc_shadow.model.norm_model import NormModel
from abc_shadow.model.two_interactions_graph_model import \
    TwoInteractionsGraphModel
from abc_shadow.model.ising_graph_model import IsingGraphModel
from abc_shadow.model.strauss_graph_model import StraussGraphModel
from abc_shadow.model.strauss_interactions_graph_model import \
    StraussInterGraphModel

ALGOS = ['abc_shadow', 'metropolis_hasting']

MODELS = ['normal',
          'binomial',
          'binomial_edge_graph',
          'binomial_graph',
          'strauss_graph',
          'strauss_inter_graph',
          'ising',
          '2_interactions']


def main():
    parser = argparse.ArgumentParser(description="Shadow Launcher")
    parser.add_argument("algo", choices=ALGOS)
    parser.add_argument("model", choices=MODELS)

    parser.add_argument("-c", "--configfile", required=True)

    arguments = parser.parse_args()

    if not os.path.isfile(arguments.configfile):
        parser.error("The file {} does not exist!".format(
            arguments.configfile))

    config = configparser.ConfigParser()
    config.read(arguments.configfile)

    if arguments.model not in config.sections():
        err = "Model is not described by the conf file"
        raise ValueError(err)

    config_model = config[arguments.model]

    theta_0 = retrieve_vector(config_model.get('theta0'))
    theta_perfect = retrieve_vector(config_model.get('thetaPerfect'))
    delta = retrieve_vector(config_model.get('delta'))
    n = config_model.getint('n')
    iters = config_model.getint('iters')
    size = config_model.getint('size')

    print('============= SUMMARY =============')
    print('theta_0: {}'.format(theta_0))
    print('theta_perfect: {}'.format(theta_perfect))
    print('delta: {}'.format(delta))
    print('n: {}'.format(n))
    print('iters: {}'.format(iters))
    print('size: {}'.format(size))

    if 'seed' in config_model:
        print("🎲  Let's make this random world determinist")
        seed = config_model.getint('seed')
        np.random.seed(seed)
        print('seed {} is enabled'.format(seed))

    if arguments.algo == 'abc_shadow':

        print("🚀 🚀 🚀 🚀  ABC SHADOW 🚀 🚀 🚀 🚀 ")

        # Default sampler
        sampler = metropolis_sampler

        if arguments.model == 'normal':
            model = NormModel(*theta_perfect)
            sampler = normal_sampler

        elif arguments.model == 'binomial':
            model = BinomialModel(*theta_perfect)
            sampler = binom_sampler

        elif arguments.model == 'binomial_edge_graph':
            model = BinomialEdgeGraphModel(*theta_perfect)

        elif arguments.model == 'binomial_graph':
            model = BinomialGraphModel(*theta_perfect)

        elif arguments.model == '2_interactions':
            model = TwoInteractionsGraphModel(*theta_perfect)

        elif arguments.model == 'ising':
            model = IsingGraphModel(*theta_perfect)

        elif arguments.model == 'strauss_graph':
            model = StraussGraphModel(*theta_perfect)

        elif arguments.model == 'strauss_inter_graph':
            model = StraussInterGraphModel(*theta_perfect)

        else:
            err = "Unknown model: {}".format(arguments.model)
            raise ValueError(err)

        print("📊 Model {} has been instanciated".format(arguments.model))

        sampler_it = config_model.getint('samplerIt')
        print("Sampler iterations: {}".format(sampler_it))

        sim_data = config_model.getboolean('simData')
        if sim_data:
            print("Observation is being generated ...")
            y_obs = sampler(model, size, 1000)
        else:
            y_obs = retrieve_vector(config_model.get('obs'))

        print("Data observed: {}".format(y_obs))

        try:
            mask = retrieve_vector(config_model.get('mask'))
            print("🎭  Mask has been set: {}".format(mask))
        except KeyError:
            mask = None

        model.set_params(*theta_0)

        start_time = time.time()
        posteriors = abc_shadow(model,
                                theta_0,
                                y_obs,
                                delta,
                                n,
                                size,
                                iters,
                                sampler=sampler,
                                sampler_it=sampler_it,
                                mask=mask)
        end_time = time.time()

        print("DURANTION : {}".format(end_time - start_time))

    elif arguments.algo == 'metropolis_hasting':
        print("🚂 🚂 🚂 Metropolis Hasting Sampling 🚂 🚂 🚂 ")

        mask = config_model.get('mask')

        if mask is not None:
            print("Mask has been set: {}".format(mask))

        if arguments.model == 'normal':
            ratio = norm_ratio
            y_obs = np.random.normal(
                theta_perfect[0], sqrt(theta_perfect[1]), size)

        elif arguments.model == 'binomial':
            ratio = binom_ratio
            n_p = theta_perfect[0]
            p_perfect = exp(theta_perfect[-1]) / (1 + exp(theta_perfect[-1]))
            p_0 = exp(theta_0[0]) / (1 + exp(theta_0[0]))

            print(p_perfect)
            theta_perfect = np.array([n_p, p_perfect])
            theta_0 = np.array([n_p, p_0])

            print("WARNING theta_perfect and theta_0 have been reassigned")
            print('theta_0: {}'.format(theta_0))
            print('theta_perfect: {}'.format(theta_perfect))

            y_obs = y_obs = np.random.binomial(
                theta_perfect[0], theta_perfect[1], size)
            mask = [1, 0]

            print("mask is forces to: {}".format(mask))
        else:
            err = "metropolis hasting cannot be used on a graph model"
            raise ValueError(err)

        print("📊 Model {} has been instanciated".format(arguments.model))
        print("Data observed: {}".format(y_obs))

        posteriors = mh_post_sampler(
            theta_0, y_obs, delta, n, iters, ratio, mask=mask)
    else:
        err = 'Given estimation algorithm {} not known'.format(arguments.algo)
        raise ValueError(err)

    print("💾  Save Record ... ")
    record = dict()
    record['algo'] = arguments.algo
    record['model'] = arguments.model
    record['theta0'] = theta_0.tolist()
    record['theta_perf'] = theta_perfect.tolist()
    record['iters'] = iters
    record['n'] = n
    record['delta'] = delta.tolist() if isinstance(
        delta, np.ndarray) else delta
    record['y_obs'] = y_obs.tolist() if isinstance(
        y_obs, np.ndarray) else y_obs
    record['posteriors'] = [post.tolist() if isinstance(post, np.ndarray)
                            else post for post in posteriors]
    timestamp = str(time.time())

    filename = '-'.join([record['algo'], record['model'], timestamp])
    filename += '.json'

    with open(filename, 'w') as output_file:
        json.dump(record, output_file)

    print("📝 Record saved in {}".format(filename))


def retrieve_vector(entry):
    if entry is None:
        err = "This entry does not exist"
        raise KeyError(err)

    vec = list(map(np.float, entry.split(',')))
    return np.array(vec)


if __name__ == "__main__":
    main()
