import numpy as np

import networkx as nx
import pytest

from abc_shadow.graph.graph_wrapper import GraphWrapper

DIM = 10
RDN_NBR = 5

"""
Fixtures
"""


@pytest.fixture(params=[DIM])
def get_graph_by_dim(request):
    graph_wrapper = GraphWrapper(request.param)
    return graph_wrapper


@pytest.fixture
def get_random_edge(request, get_graph_by_dim):
    edges = list(get_graph_by_dim.get_elements())
    random_edge = edges[np.random.choice(len(edges))]
    return random_edge


@pytest.fixture(params=[RDN_NBR])
def get_random_edges(request, get_graph_by_dim):
    edges = list(get_graph_by_dim.get_elements())
    random_indices = np.random.choice(len(edges), replace=False, size=RDN_NBR)
    return [edges[i] for i in random_indices]


@pytest.fixture
def get_florentine_graph(request):
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
    e = get_random_edge
    none_edge_count = get_graph_by_dim.get_none_edge_count()
    get_graph_by_dim.set_edge_type(e, 1)

    assert get_graph_by_dim.get_edge_type(e) == 1
    assert get_graph_by_dim.get_edge_count() == 1
    assert get_graph_by_dim.get_none_edge_count() == none_edge_count - 1


def test_is_enabled_edge(get_graph_by_dim, get_random_edge):
    e = get_random_edge
    get_graph_by_dim.set_edge_type(e, 0)
    assert not get_graph_by_dim.is_active_edge(e)

    get_graph_by_dim.set_edge_type(e, 1)
    assert get_graph_by_dim.is_active_edge(e)


def test_get_enabled_edges(get_graph_by_dim, get_random_edges):
    edges = get_random_edges
    for e in edges:
        get_graph_by_dim.set_edge_type(e, 1)

    enabled_edges = get_graph_by_dim.get_enabled_edges()

    assert len(enabled_edges) == len(edges)
    assert sorted(enabled_edges) == sorted(edges)
