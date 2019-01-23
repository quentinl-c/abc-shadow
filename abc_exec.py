from abc_shadow.model.binomial_graph_model import BinomialGraphModel
from abc_shadow.sample_generator import (generate_bernouilli_stats,
                                         generate_norm_stats)
from abc_shadow.model.norm_model import NormModel
from abc_shadow.exec_reports import ABCExecutionReport, ABCExecutionReportJSON
from abc_shadow.abc_impl import abc_shadow
import json
import numpy as np


def main():
    theta_perfect = np.array([2, 9])
    theta_0 = np.array([1.6, 8.7])
    model = NormModel(theta_0[0], theta_0[1])
    # model = BernouilliModel(*theta_perfect)
    size = 20
    it = 10000
    y = generate_bernouilli_stats(size, it, theta_perfect[0], theta_perfect[1])

    delta = np.array([0.005, 0.005])
    iters = 1000
    n = 10

    posteriors = abc_shadow(model,
                            theta_0,
                            y,
                            delta,
                            n,
                            size,
                            iters,
                            sampler_it=1000)

    json_list = [post.tolist() if isinstance(post, np.ndarray)
                 else post for post in posteriors]

    with open('posterior10.json', 'w') as output_file:
        output_file.truncate()
        json.dump(json_list, output_file)

    print("END !")

if __name__ == '__main__':
    main()
