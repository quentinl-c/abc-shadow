import json
from abc_shadow.model.markov_star_graph_model import MarkovStarGraphModel
from abc_shadow.graph.graph_wrapper import GraphWrapper
from abc_shadow.sampler import mcmc_sampler
import numpy as np

path = './results/Markov_star_graph/assets/abc_shadow-markov_2star_graph-1544956169.1947615.json'
with open(path, 'r') as input_file:
    data = json.load(input_file)
    if not isinstance(data, list):
        data = data['posteriors']

m = MarkovStarGraphModel(*data[0])
g = GraphWrapper(16)

np.random.seed(2018)
for p in data[::100]:
    print(p)
    res = mcmc_sampler(g, m, iters=1000)

    # print(m.summary_dict(res))
    stats = [np.mean(s) for s in m.summary_dict(res).values()]
    print("Edge count : {} 2 star Counts : {} ".format(*stats))
    print("U = {}".format(m.evaluate_from_stats(*stats)))
