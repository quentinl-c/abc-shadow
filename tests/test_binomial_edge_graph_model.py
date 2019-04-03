import numpy as np

import pytest

from abc_shadow.model.binomial_edge_graph_model import BinomialEdgeGraphModel


def test_init():
    edge_param_expected = np.random.uniform(0, 100)
    model = BinomialEdgeGraphModel(edge_param_expected)
    edge_param = model.edge_param
    assert edge_param == edge_param_expected


@pytest.mark.usefixtures("get_empty_binomial_edge_model")
def tes_edge_param(get_empty_binomial_edge_model):
    model = get_empty_binomial_edge_model
    edge_param_expected = np.random.uniform(0, 100)

    model.edge_param = edge_param_expected

    edge_param = model.edge_param
    assert edge_param == edge_param_expected


@pytest.mark.usefixtures("get_empty_binomial_edge_model")
def test_set_params(get_empty_binomial_edge_model):
    model = get_empty_binomial_edge_model

    with pytest.raises(ValueError):
        model.set_params()
    edge_param_expected = np.random.uniform(0, 100)

    model.set_params(edge_param_expected)

    edge_param = model.edge_param
    assert edge_param == edge_param_expected


def test_evaluate_from_stats():
    edge_param = np.random.uniform(0, 100)
    model = BinomialEdgeGraphModel(edge_param)

    with pytest.raises(ValueError):
        model.evaluate_from_stats()

    stats = np.random.uniform(0, 100)
    eval_m = model.evaluate_from_stats(stats)

    eval_m_expected = edge_param * stats

    assert eval_m == eval_m_expected


@pytest.mark.usefixtures("get_one_edge_enabled_graph")
def test_evaluate(get_one_edge_enabled_graph):
    sample = get_one_edge_enabled_graph
    edge_param = np.random.uniform(0, 100)
    model = BinomialEdgeGraphModel(edge_param)

    eval_m = model.evaluate(sample)
    eval_m_expected = edge_param

    assert eval_m == eval_m_expected


@pytest.mark.usefixtures("get_empty_graph", "get_random_edge")
def test_get_local_energy(get_empty_graph, get_random_edge):
    sample = get_empty_graph

    edge_param = np.random.uniform(0, 100)
    model = BinomialEdgeGraphModel(edge_param)
    e = get_random_edge

    local_u_before = model.get_local_energy(sample, e)
    local_u_before_expected = 0

    assert local_u_before == local_u_before_expected

    sample.set_edge_type(e, 1)

    local_u_after = model.get_local_energy(sample, e)
    local_u_after_expected = edge_param

    assert local_u_after == local_u_after_expected


@pytest.mark.usefixtures("get_empty_graph", "get_random_edge")
def test_compute_delta_stats(get_empty_graph, get_random_edge):
    sample = get_empty_graph
    e = get_random_edge

    model = BinomialEdgeGraphModel(0)
    delta_stats = model.get_delta_stats(sample, e, 1)

    expected_delta_stats = np.array([1])

    assert delta_stats - expected_delta_stats