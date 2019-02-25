import pytest
import numpy as np
from abc_shadow.model.strauss_interactions_graph_model import \
    StraussInterGraphModel


@pytest.mark.usefixture("get_empty_graph")
def test_compute_delta(get_empty_graph):
    params = np.random.uniform(0, 100, size=3)
    model = StraussInterGraphModel(*params)
    gr = get_empty_graph

    ego = list(gr.vertex.keys())[np.random.choice(len(gr.vertex.keys()))]
    labels = np.random.randint(low=0, high=3, size=len(gr.graph[ego]))

    for i in range(len(labels)):
        gr.set_edge_type(gr.graph[ego][i], labels[i])

    given_delta = model.compute_delta(gr, ego, 0)
    u_before = model.evaluate(gr)
    given_delta = model.compute_delta(gr, ego, 1)
    u_after = model.evaluate(gr)
    assert pytest.approx(given_delta - u_after + u_before,
                         rel=1e-6, abs=1e-8) == 0
