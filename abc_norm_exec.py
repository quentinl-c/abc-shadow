from abc_shadow.model.norm_model import NormModel
from abc_shadow.exec_reports import (ABCExecutionReport,
                                             ABCExecutionReportJSON)
from abc_shadow.abc_impl import abc_shadow, normal_sampler
import json
import numpy as np


def main():
    theta_prior = np.array([10, 10])
    model = NormModel(theta_prior[0], theta_prior[1])
    seed = 2018
    size = 1000
    mean = 2
    std = 3
    np.random.seed(2018)
    sample_obs = np.random.normal(mean, std, size)
    y_obs = [np.average(stats)
             for stats in model.summary([sample_obs]).values()]
    delta = np.array([0.05, 0.05])
    iters = 2000
    n = 100

    posteriors = abc_shadow(model,
                            theta_prior,
                            y_obs,
                            delta,
                            n,
                            size,
                            iters,
                            sampler=normal_sampler)

    json_list = [post.tolist() if isinstance(post, np.ndarray)
                 else post for post in posteriors]

    with open('norm.json', 'w') as output_file:
        output_file.truncate()
        json.dump(json_list, output_file)

    print("END !")


if __name__ == '__main__':
    main()
