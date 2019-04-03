from abc_shadow.graph.graph_wrapper import GraphWrapper
from abc_shadow.sampler import mcmc_sampler
from abc_shadow.model.binomial_graph_model import BinomialGraphModel
from abc_shadow.model.binomial_edge_graph_model import BinomialEdgeGraphModel
# from abc_shadow.utils import dist_display, display
from abc_shadow.abc_impl import binom_graph_sampler
from abc_shadow.model.potts_graph_model import PottsGraphModel
import numpy as np
import time
import timeit
from matplotlib import pyplot as plt
import scipy.stats as stats


def cumul_avg(tab):
    """Given a list of computable values (int/float)
    computes the corresponding list of moving average
    (cumulative average)

    Arguments:
        tab {List[int]} -- List of observed values

    Returns:
        {List[int]} -- Corresponding list of moving average
    """

    res = list()

    for i in range(1, len(tab)):
        cum_avg = sum(tab[0:i]) / (i + 1)
        res.append(cum_avg)

    return res


def cdf(data):
    """ CDF : Cumulative Distribution function
    Returns the Cumulative Distribution of the dataset (data)

    Arguments:
        data {List[float]} -- dataset

    Returns:
        {List[foat], List[float]} -- absciss, ordinate
    """

    x = np.sort(data)
    y = np.arange(1, len(data) + 1) / len(data)
    return x, y


def display(dataset,  title, prefix=""):
    """Creates a fil displaying a summary of the model

    Arguments:
        results {List[GraphWrapper)] -- instance of the graph
                                        for each iteration
        summary_fct {fonction} -- function returning a summary dict
        title {str} -- Figure's title

    Keyword Arguments:
        prefix {str} -- filen (default: {""})
    """
    """
    Displays five different charts

    Arguments:
        results {[type]} - - [description]

    Keyword Arguments:
        prefix {str} - - [description](default: {""})
    """

    fig = plt.figure(figsize=(10, 10))

    for idx, label in enumerate(dataset):
        ax = plt.subplot(len(dataset), 1, idx + 1)
        ax.plot(dataset[label], label='Sample value')
        ax.plot(cumul_avg(dataset[label]), label='Cumulative average')
        ax.set_ylabel(label)

    plt.legend()
    plt.xlabel("Iterations")

    plt.subplots_adjust(hspace=0.4)
    plt.suptitle('Metropolis hasting algorithm : Markov model')

    fig.savefig(prefix + "chain.pdf")


def dist_display(dataset, prefix=""):
    """Displays the samples’ distribution.

    Arguments:
        results {List[GraphWrapper]} -- List of resulting graph
                                        for each iteration

        summary_fct {fonction} -- function returning a summary dict

    Keyword Arguments:
        prefix {str} -- filename prefix (default: {""})
    """

    fig = plt.figure(figsize=(20, 20))
    fig.suptitle('Distribution Summary', fontsize=30)
    # Creates a grid with as many rows as entries in dataset and 2 columns
    # The first column is dedicated to the histogramme
    # The second one is dedicated to the CDF
    grid = plt.GridSpec(len(dataset), 3, hspace=0.6)

    for idx, label in enumerate(dataset):
        # Iterates over each grid's line
        hist_area = fig.add_subplot(grid[idx, 0])
        cdf_area = fig.add_subplot(grid[idx, 1])
        qq_plot_area = fig.add_subplot(grid[idx, 2])

        hist_area.set_title('{} Distribution'.format(label))
        hist_area.set_xlabel('Observed value')
        hist_area.set_ylabel('Distribution density')
        hist_area.hist(dataset[label])

        cdf_area.set_title('{}\' CDF'.format(label))
        cdf_area.set_xlabel('Observed value')
        cdf_area.set_ylabel('Density')
        cdf_x, cdf_y = cdf(dataset[label])
        cdf_area.plot(cdf_x, cdf_y)

        stats.probplot(dataset[label], dist="norm", plot=qq_plot_area)
        qq_plot_area.set_title("Normal Q-Q plot for {}".format(label))
        qq_plot_area.get_lines()[0].set_markerfacecolor('C0')
        qq_plot_area.get_lines()[1].set_color('C2')

    fig.savefig(prefix + "dist.pdf")

def main():
    # m = BinomialGraphModel(8, 0)
    np.random.seed(2019)

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
    g = GraphWrapper(30)

    # for i in range(1):
    #     print(i)
    iters = 500

    res = mcmc_sampler(g, m, iters=iters)
        # print(m.summary_dict(res))

    # summary = [np.mean(s) for s in m.summary_dict(res).values()]
    summary = np.mean(res, axis=0)

    # with open('sim-map.txt', 'w') as res_file:
    #     for it in range(iters - 1):
    #         line = " ".join([str(summary[0][it]), str(summary[1][it]), str(summary[2][it])])
    #         res_file.writelines(line + "\n")

    # means = np.mean([s for s in m.summary_dict(res).values()], axis=1)
    #     # print(means)
    print(time.time() - start)

    # dist_display(res, m, prefix='hist__{}__'.format(0))

    # display(res, m, "Model", prefix='__{}__'.format(0))

    # print(means)
    print(summary)
    # print(res)
    data = np.array(res).transpose()
    print(data)

    dataset = dict()
    dataset['beta01'] = data[0,:]
    dataset['beta02'] = data[1,:]
    dataset['beta12'] = data[2,:]

    dist_display(dataset)
    display(dataset, "potts")
    # # print(" None edge count : {}; Edge count : {}".format(np.mean(stats['None edges counts']), np.mean(stats['Edges counts'])))

    # m = BinomialGraphModel(0.5, 1)

    # print(binom_graph_sampler(m, 10, 100))
    # dist_display(res, m, prefix='hist_')
    # edge_dyad_display(res, prefix='_')
    # display(res, m, "Binomial Model", prefix='_')
    # qq_plot(res, prefix='qqplot')

if __name__ == '__main__':
    # timeit.timeit('main()', number=10)
    # print(timeit.Timer('main()', number=1).timeit())
    main()
