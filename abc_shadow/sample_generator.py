from .graph.graph_wrapper import GraphWrapper
import copy
import numpy as np
from .model.bernouilli_model import BernouilliModel
import math


def generate_bernouilli_stats(dim, iters, edge_param, none_edge_param):
    sample = GraphWrapper(dim)
    none_edge_prob = none_edge_param / (edge_param + none_edge_param)
    edge_prob = edge_param / (edge_param + none_edge_param)
    probs = [none_edge_prob, edge_prob]

    dist = list()

    for _ in range(iters):
        for edge in sample.get_elements():
            edge_attr = np.random.choice(a=list(BernouilliModel.type_values),
                                         size=1,
                                         replace=False,
                                         p=probs)
            sample.set_edge_type(edge, edge_attr[0])
        dist.append(copy.deepcopy(sample))

    mean_stats = (np.mean([s.get_none_edge_count() for s in dist]),
                  np.mean([s.get_edge_count() for s in dist]))
    return mean_stats