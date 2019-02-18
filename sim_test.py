from abc_shadow.graph.graph_wrapper import GraphWrapper
from abc_shadow.sampler import mcmc_sampler
from abc_shadow.model.binomial_graph_model import BinomialGraphModel
from abc_shadow.model.binomial_edge_graph_model import BinomialEdgeGraphModel
from abc_shadow.utils import dist_display, display
from abc_shadow.abc_impl import binom_graph_sampler
from abc_shadow.model.two_interactions_graph_model import \
    TwoInteractionsGraphModel
from abc_shadow.model.ising_graph_model import IsingGraphModel
import numpy as np
import time


def main():
    # m = BinomialGraphModel(8, 0)
    np.random.seed(2018)

    # m = BinomialEdgeGraphModel(-2.197224577336219)
    m = TwoInteractionsGraphModel(-4.2, 1, 1, 0.6)
    # m = IsingGraphModel(0.5, 0.5, 0.5)
    # for _ in range(10):

    g = GraphWrapper(10)
    stats = list()
    # for _ in range(100):
    # start = time.time()
    for i in range(1):
        print(i)
        res = mcmc_sampler(g, m, iters=1000)
        # print(m.summary(res))
        means = np.mean([s for s in m.summary(res).values()], axis=1)
        print(means)
        stats.append(means)

        # dist_display(res, m, prefix='hist_{}_'.format(i))

        # display(res, m, "Model", prefix='_{}_'.format(i))

    print(np.mean(stats, axis=0))
    print(np.std(stats, axis=0))
    # print(time.time() - start)


    # # print(" None edge count : {}; Edge count : {}".format(np.mean(stats['None edges counts']), np.mean(stats['Edges counts'])))

    # m = BinomialGraphModel(0.5, 1)

    # print(binom_graph_sampler(m, 10, 100))
    # dist_display(res, m, prefix='hist_')
    # edge_dyad_display(res, prefix='_')
    # display(res, m, "Binomial Model", prefix='_')
    # qq_plot(res, prefix='qqplot')

if __name__ == '__main__':
    main()
