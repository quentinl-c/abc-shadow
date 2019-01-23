from abc_shadow.graph.graph_wrapper import GraphWrapper
from abc_shadow.sampler import mcmc_sampler
from abc_shadow.model.binomial_graph_model import BinomialGraphModel
from abc_shadow.model.binomial_edge_graph_model import BinomialEdgeGraphModel
from abc_shadow.model.markov_star_graph_model import MarkovStarGraphModel
from abc_shadow.utils import dist_display, display
from abc_shadow.abc_impl import binom_graph_sampler
from abc_shadow.model.two_interactions_graph_model import TwoInteractionsGraphModel
import numpy as np
import time


def main():
    # m = BinomialGraphModel(8, 0)
    np.random.seed(2018)

    # m = BinomialEdgeGraphModel(-2.197224577336219)
    m = TwoInteractionsGraphModel(2.09275698,  1.87109988, -0.04888886)
    # m = MarkovStarGraphModel(-1.649619, 0.007925)
    g = GraphWrapper(10)

    start = time.time()
    res = mcmc_sampler(g, m, iters=1000)

    print(time.time() - start)
    # print(m.summary(res))
    stats = [np.mean(s) for s in m.summary(res).values()]

    print(stats)

    # # print(" None edge count : {}; Edge count : {}".format(np.mean(stats['None edges counts']), np.mean(stats['Edges counts'])))

    # m = BinomialGraphModel(0.5, 1)

    # print(binom_graph_sampler(m, 10, 100))
    #dist_display(res, m, prefix='hist_')
    # edge_dyad_display(res, prefix='_')
    #display(res, m, "Binomial Model", prefix='_')
    # qq_plot(res, prefix='qqplot')

if __name__ == '__main__':
    main()
