import numpy as np

from collections import Counter
import pytest
from abc_shadow.model.two_interactions_graph_model import \
    TwoInteractionsGraphModel


def test_init():
    l0_expected, l1_expected, l2_expected, l12_expected = np.random.uniform(
        0, 100, size=4)
    model = TwoInteractionsGraphModel(l0_expected,
                                      l1_expected,
                                      l2_expected,
                                      l12_expected)

    assert model.l0 == l0_expected
    assert model.l1 == l1_expected
    assert model.l2 == l2_expected
    assert model.l12 == l12_expected


def test_set_params():
    model = TwoInteractionsGraphModel(*[0]*4)
    expected_params = np.random.uniform(
        0, 100, size=4)
    model.set_params(*expected_params)
    assert model.l0 == expected_params[0]
    assert model.l1 == expected_params[1]
    assert model.l2 == expected_params[2]
    assert model.l12 == expected_params[3]


def test_evaluate_from_stats():
    params = np.random.uniform(0, 100, size=4)
    model = TwoInteractionsGraphModel(*params)
    stats = np.random.uniform(0, 100, size=4)
    u = model.evaluate_from_stats(*stats)
    assert u == np.dot(params, stats)


@pytest.mark.usefixture("get_empty_graph")
def test_get_local_energy(get_empty_graph):
    params = np.random.uniform(0, 100, size=4)
    model = TwoInteractionsGraphModel(*params)
    gr = get_empty_graph

    ego = list(gr.vertex.keys())[np.random.choice(len(gr.vertex.keys()))]
    labels = np.random.randint(low=0, high=3, size=len(gr.graph[ego]))
    summary = Counter(labels)

    for i in range(len(labels)):
        gr.set_edge_type(gr.graph[ego][i], labels[i])

    # edge is set to 0
    gr.set_edge_type(ego, 0)
    expected_local_u = 0
    given_u = model.get_local_energy(gr, ego, gr.graph[ego])
    assert expected_local_u == given_u

    # edge is set to 1
    gr.set_edge_type(ego, 1)
    expected_local_u = params[0] + params[1] + summary[2] * params[3]
    given_u = model.get_local_energy(gr, ego, gr.graph[ego])
    assert expected_local_u == given_u

    # edge is set to 2
    gr.set_edge_type(ego, 2)
    expected_local_u = params[0] + params[2] + summary[1] * params[3]
    given_u = model.get_local_energy(gr, ego, gr.graph[ego])
    assert expected_local_u == given_u

    model.set_params(0, 0, 0, 1)
    expected_local_u = gr.get_local_repulsion_count(ego, excluded_labels=[0])
    given_u = model.get_local_energy(gr, ego, gr.graph[ego])
    assert expected_local_u == given_u


@pytest.mark.usefixture("get_empty_graph")
def test_compute_delta(get_empty_graph):
    params = np.random.uniform(0, 100, size=4)
    model = TwoInteractionsGraphModel(*params)
    gr = get_empty_graph

    ego = list(gr.vertex.keys())[np.random.choice(len(gr.vertex.keys()))]
    labels = np.random.randint(low=0, high=3, size=len(gr.graph[ego]))
    summary = Counter(labels)

    for i in range(len(labels)):
        gr.set_edge_type(gr.graph[ego][i], labels[i])

    # Move 0 -> 1
    gr.set_edge_type(ego, 0)
    expected_delta = params[0] + params[1] + summary[2] * params[3]
    given_delta = model.compute_delta(gr, ego, 1)
    assert expected_delta == given_delta

    # move 1 -> 2
    gr.set_edge_type(ego, 1)
    expected_delta = params[0] + params[2] + summary[1] * \
        params[3] - (params[0] + params[1] + summary[2] * params[3])
    given_delta = model.compute_delta(gr, ego, 2)
    assert expected_delta == given_delta
