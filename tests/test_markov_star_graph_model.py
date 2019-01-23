import numpy as np

import pytest

from abc_shadow.model.markov_star_graph_model import MarkovStarGraphModel


def test_init():
    edge_param_expected, two_star_param_expected = np.random.uniform(
        0, 100, size=2)

    model = MarkovStarGraphModel(edge_param_expected, two_star_param_expected)

    edge_param = model.edge_param
    assert edge_param == edge_param_expected

    two_star_param = model.get_two_star_param()
    assert two_star_param == two_star_param_expected


@pytest.mark.usefixtures("get_empty_markov_star_model")
def tes_edge_param(get_empty_markov_star_model):
    model = get_empty_markov_star_model
    edge_param_expected = np.random.uniform(0, 100)

    model.edge_param = edge_param_expected

    edge_param = model.edge_param
    assert edge_param == edge_param_expected


@pytest.mark.usefixtures("get_empty_markov_star_model")
def test_set_two_star_param(get_empty_markov_star_model):
    model = get_empty_markov_star_model
    two_star_param_expected = np.random.uniform(0, 100)

    model.set_two_star_param(two_star_param_expected)

    two_star_param = model.get_two_star_param()
    assert two_star_param == two_star_param_expected


@pytest.mark.usefixtures("get_empty_markov_star_model")
def test_set_params(get_empty_markov_star_model):
    model = get_empty_markov_star_model

    with pytest.raises(ValueError):
        model.set_params(42)

    edge_param_expected, two_star_param_expected = np.random.uniform(
        0, 100, size=2)

    model.set_params(edge_param_expected, two_star_param_expected)

    edge_param = model.edge_param
    assert edge_param == edge_param_expected

    two_star_param = model.get_two_star_param()
    assert two_star_param == two_star_param_expected


def test_evaluate_from_stats():
    edge_param, two_star_param = np.random.uniform(0, 100, size=2)

    model = MarkovStarGraphModel(edge_param, two_star_param)

    with pytest.raises(ValueError):
        model.evaluate_from_stats()

    e_stats, s_stats = np.random.uniform(0, 100, size=2)
    eval_m = model.evaluate_from_stats(e_stats, s_stats)

    eval_m_expected = edge_param * e_stats + two_star_param * s_stats

    assert eval_m == eval_m_expected


@pytest.mark.usefixtures("get_one_star_enabled_graph")
def test_evaluate(get_one_star_enabled_graph):
    sample = get_one_star_enabled_graph
    edge_param, two_star_param = np.random.uniform(0, 100, size=2)

    model = MarkovStarGraphModel(edge_param, two_star_param)

    eval_m = model.evaluate(sample)
    eval_m_expected = edge_param * 2 + two_star_param

    assert eval_m == eval_m_expected


@pytest.mark.usefixtures("get_empty_graph", "get_random_edge")
def test_get_local_energy(get_empty_graph, get_random_edge):
    sample = get_empty_graph

    edge_param, two_star_param = np.random.uniform(0, 100, size=2)
    model = MarkovStarGraphModel(edge_param, two_star_param)
    e = get_random_edge

    local_u_0 = model.get_local_energy(sample, e)
    local_u_0_expected = 0

    assert local_u_0 == local_u_0_expected

    sample.set_edge_type(e, 1)

    local_u_after = model.get_local_energy(sample, e)
    local_u_after_expected = edge_param

    assert local_u_after == local_u_after_expected


@pytest.mark.usefixtures("get_empty_graph", "get_random_edge")
def test_compute_delta(get_empty_graph, get_random_edge):
    sample = get_empty_graph

    edge_param, two_star_param = np.random.uniform(0, 100, size=2)
    model = MarkovStarGraphModel(edge_param, two_star_param)
    e = get_random_edge
    neigh = next(iter(sample.graph.neighbors(e)))
    sample.set_edge_type(neigh, 1)

    delta_expected = edge_param + two_star_param
    delta = model.compute_delta(sample, e, 1)

    assert delta == delta_expected
