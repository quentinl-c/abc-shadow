import random

import networkx as nx
import pytest

from abc_shadow.graph.graph_wrapper import GraphWrapper

DIM = 10

"""
Fixtures
"""


@pytest.fixture(params=[DIM])
def get_graph_by_dim(request):
    graph_wrapper = GraphWrapper(request.param)
    return graph_wrapper


@pytest.fixture
def get_random_edge(get_graph_by_dim):
    edges = get_graph_by_dim.get_elements()
    random_edge = random.sample(list(iter(edges)), k=1)[0]
    return random_edge


@pytest.fixture
def get_florentine_graph():
    return nx.florentine_families_graph()

"""
Test cases
"""


def test_graph_creation_from_graph(get_florentine_graph):
    graphwrapper = GraphWrapper(get_florentine_graph)
    nx.set_edge_attributes(get_florentine_graph, 1, 'type')

    compl_flor = nx.complement(get_florentine_graph)
    get_florentine_graph.add_edges_from(compl_flor.edges(), type=0)

    edge_attr = set({tuple(sorted(key)): int(val)
                 for key, val in nx.get_edge_attributes(get_florentine_graph, 'type').items()})

    node_attr = set(nx.get_node_attributes(graphwrapper.get_graph(), 'type'))
    assert edge_attr == node_attr


def test_graph_dim(get_graph_by_dim):
    assert get_graph_by_dim.get_initial_dim() == DIM


def test_edge_attr(get_graph_by_dim, get_random_edge):
    none_edge_count = get_graph_by_dim.get_none_edge_count()
    get_graph_by_dim.set_edge_type(get_random_edge, 1)

    assert get_graph_by_dim.get_edge_type(get_random_edge) == 1
    assert get_graph_by_dim.get_edge_count() == 1
    assert get_graph_by_dim.get_none_edge_count() == none_edge_count - 1


def test_dyad_edge_attr(get_graph_by_dim, get_random_edge):
    none_edge_count = get_graph_by_dim.get_none_edge_count()
    get_graph_by_dim.set_edge_type(get_random_edge, 2)

    assert get_graph_by_dim.get_edge_type(get_random_edge) == 2
    assert get_graph_by_dim.get_dyadic_count() == 1
    assert get_graph_by_dim.get_none_edge_count() == none_edge_count - 1
