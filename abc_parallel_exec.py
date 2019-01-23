import threading
from itertools import product
from multiprocessing import JoinableQueue, Process, cpu_count, current_process
import json
import numpy as np
from sortedcontainers import SortedList

from abc_shadow.model.binomial_graph_model import BinomialGraphModel
from abc_shadow.sample_generator import generate_bernouilli_stats
from abc_shadow.exec_reports import (ABCExecutionReport,
                                     ABCExecutionReportJSON)
from abc_shadow.abc_impl import abc_shadow

import time
THREAD_NBR = cpu_count()


def worker(in_queue, out_queue):
    name = current_process().name
    print("===🚀 worker {} is starting 🚀===".format(name))

    while True:
        exec_report = in_queue.get()
        exec_report.posteriors = abc_shadow(exec_report.model,
                                            exec_report.theta_0,
                                            exec_report.y,
                                            exec_report.delta,
                                            exec_report.n,
                                            exec_report.dim,
                                            exec_report.iters,
                                            sampler_it=(
                                                exec_report.mh_sampler_iter))
        if exec_report.posteriors is not None:
            out_queue.put(exec_report)
        in_queue.task_done()


def reviewer(out_queue, in_queue, theta_perfect):
    print("🔍 Reviewer is starting 🔍")
    bound = 20

    res_last_sort = SortedList(key=lambda x: x[0])
    timestamp = str(time.time())

    while True:
        exec_report = out_queue.get()

        last_candidate = exec_report.posteriors[-1]
        last_diff = np.linalg.norm(theta_perfect - last_candidate)
        res_last_sort.add((last_diff, exec_report))
        res_last_sort = SortedList(res_last_sort[:bound], key=lambda x: x[0])

        print("💾 Backup {} bestof  💾 ".format(bound))
        with open('best-last-candidates-{}.json'.format(timestamp), 'w') as output_file:
            output_file.truncate()
            json.dump([ABCExecutionReportJSON().default(res[1]) for res in res_last_sort], output_file)

        out_queue.task_done()


def print_remaining_executions(in_queue):
    try:
        msg = "Remaining executions {}".format(in_queue.qsize())

    except NotImplementedError:
        msg = "Sorry you are actually using macOS :"\
              "you can't monitor queue size 🍏 💻 🍎"

    print(msg)
    if not in_queue.empty():
        th = threading.Timer(600, print_remaining_executions, args=(in_queue,))
        th.start()


def main():

    theta_perfect = np.array([1, 1])

    theta_0 = np.array([1, 0.9])
    model = BinomialGraphModel(*theta_perfect)
    dim = 10
    print("=== Generate statistic observation ⏳ ⏳ ===")
    it = 1000
    y = generate_bernouilli_stats(dim, it, theta_perfect[0], theta_perfect[1])

    in_queue = JoinableQueue()
    out_queue = JoinableQueue()

    workers = list()

    print("====🤖 {} worker are launching ...🤖 ====".format(THREAD_NBR - 1))

    for _ in range(THREAD_NBR - 1):
        p = Process(target=worker, args=(in_queue, out_queue))
        p.daemon = True
        p.start()
        workers.append(p)

    # delta_range = np.arange(0.005, 0.015, 0.005)
    delta_range = [0.005, 0.01]
    n_range = [10, 20]
    iters_range = [5000, 10000, 100000]
    mh_sampler_iter_range = [50, 100]

    combinations = product(product(delta_range, repeat=2),
                           n_range, iters_range, mh_sampler_iter_range)

    for param in combinations:
        delta, n, iters, mh_sampler_iter = param
        exec_report = ABCExecutionReport(
            delta, n, iters, model, theta_0, y, dim, mh_sampler_iter)
        in_queue.put(exec_report)
    # in_queue.put(ABCExecutionReport((0.005, 0.005), 10, 10, model,
    #                                 theta_0, y, dim, 20))
    # in_queue.put(ABCExecutionReport((0.005, 0.005), 10, 10, model,
    #                                 theta_0, y, dim, 20))
    print_remaining_executions(in_queue)

    reviewer_p = Process(target=reviewer, args=(
        out_queue, in_queue, theta_perfect))
    reviewer_p.daemon = True
    reviewer_p.start()

    in_queue.join()
    out_queue.join()

if __name__ == '__main__':
    main()
