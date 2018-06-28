from metropolis_hasting.graph.graph_wrapper import GraphWrapper
from metropolis_hasting.metropolis_hasting import metropolis_hasting
from metropolis_hasting.model.dyadic_model import DyadicModel
from metropolis_hasting.util import (dyad_summary,
                                     display,
                                     markov_summary,
                                     dist_display)
from metropolis_hasting.model.markov_model import MarkovModel


def main():
    m = MarkovModel(0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
    # m = DyadicModel(0.5, 0.5, 0.5)
    g = GraphWrapper(20)

    res = metropolis_hasting(g, m, iter=100)
    dist_display(res, markov_summary, prefix='hist_')
    # edge_dyad_display(res, prefix='_')
    display(res, markov_summary, "Markov Model", prefix='_')
    # qq_plot(res, prefix='qqplot')

if __name__ == '__main__':
    main()
