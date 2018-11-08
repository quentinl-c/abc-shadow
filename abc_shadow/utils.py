from matplotlib import pyplot as plt
import matplotlib as mpl
import scipy.stats as stats
import numpy as np

mpl.style.use('seaborn')


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


def display(results,  model, title, prefix=""):
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

    dataset = model.summary(results)
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

    fig.savefig(prefix + "res.pdf")


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


def dist_display(results, model, prefix=""):
    """Displays the samples’ distribution.

    Arguments:
        results {List[GraphWrapper]} -- List of resulting graph
                                        for each iteration

        summary_fct {fonction} -- function returning a summary dict

    Keyword Arguments:
        prefix {str} -- filename prefix (default: {""})
    """
    dataset = model.summary(results)

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

    fig.savefig(prefix + "res.pdf")
