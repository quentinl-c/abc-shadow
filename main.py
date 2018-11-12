from abc_shadow.graph.graph_wrapper import GraphWrapper
from abc_shadow.sampler import mh_sampler
from abc_shadow.model.binomial_graph_model import BinomialGraphModel
from abc_shadow.utils import dist_display, display
from abc_shadow.abc_impl import binom_graph_sampler
import numpy as np


def main():
    m = BinomialGraphModel(1,5)
    g = GraphWrapper(10)

    res = mh_sampler(g, m, iters=100)
    stats = m.summary(res)
    print(" None edge count : {}; Edge count : {}".format(np.mean(stats['None edges counts']), np.mean(stats['Edges counts'])))
    
    print(binom_graph_sampler(m, 10, 100))
    #dist_display(res, m, prefix='hist_')
    # edge_dyad_display(res, prefix='_')
    #display(res, m, "Binomial Model", prefix='_')
    # qq_plot(res, prefix='qqplot')

if __name__ == '__main__':
    main()
