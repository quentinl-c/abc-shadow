from abc_shadow.graph.graph_wrapper import GraphWrapper
from abc_shadow.sampler import mcmc_sampler
from abc_shadow.model.binomial_graph_model import BinomialGraphModel
from abc_shadow.model.binomial_edge_graph_model import BinomialEdgeGraphModel
from abc_shadow.utils import dist_display, display
from abc_shadow.abc_impl import binom_graph_sampler
from abc_shadow.model.potts_graph_model import PottsGraphModel
import numpy as np
import time


def main():
    # m = BinomialGraphModel(8, 0)
    np.random.seed(2018)

    # m = BinomialEdgeGraphModel(-2.197224577336219)
    m = PottsGraphModel(0.535, -0.06, 0.535)
    # m = StraussInterGraphModel(0.535, -0.06, 0.535)
    # m = StraussGraphModel(12, 10, 12, 1)
    # m = StraussGraphModel2(1, 4.8, 0.5)
    # m = StraussGraphModel(-0.0680136,   0.39364956,  0.06566127,  1.13095765)
    # m = TwoInteractionsGraphModel()
    # m = IsingGraphModel(0.5, 0.5, 0.5)
    # for _ in range(10):

    # stats = list()
    # for _ in range(100):
    start = time.time()
    g = GraphWrapper(10)

    # for i in range(1):
    #     print(i)
    iters = 10000

    res = mcmc_sampler(g, m, iters=iters)
        # print(m.summary(res))

    summary = [s for s in m.summary(res).values()]
    with open('sim-map.txt', 'w') as res_file:
        for it in range(iters - 1):
            line = " ".join([str(summary[0][it]), str(summary[1][it]), str(summary[2][it])])
            res_file.writelines(line + "\n")

    means = np.mean([s for s in m.summary(res).values()], axis=1)
        # print(means)
    print(time.time() - start)

    dist_display(res, m, prefix='hist__{}__'.format(0))

    display(res, m, "Model", prefix='__{}__'.format(0))

    print(means)


    # # print(" None edge count : {}; Edge count : {}".format(np.mean(stats['None edges counts']), np.mean(stats['Edges counts'])))

    # m = BinomialGraphModel(0.5, 1)

    # print(binom_graph_sampler(m, 10, 100))
    # dist_display(res, m, prefix='hist_')
    # edge_dyad_display(res, prefix='_')
    # display(res, m, "Binomial Model", prefix='_')
    # qq_plot(res, prefix='qqplot')

if __name__ == '__main__':
    main()
