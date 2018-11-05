import random

import networkx as nx
import pytest

from abc_shadow.graph.digraph_wrapper import (DiGraphWrapper,
                                              _lg_directed_in_in,
                                              _lg_directed_out_out)

DIM = 10

"""
Fixtures
"""


@pytest.fixture(params=[DIM])
def get_graph_by_dim(request):
    graph_wrapper = DiGraphWrapper(request.param)
    return graph_wrapper


@pytest.fixture
def get_random_edge(get_graph_by_dim):
    edges = get_graph_by_dim.get_elements()
    random_edge = random.sample(list(edges), k=1)[0]
    return random_edge


@pytest.fixture
def get_florentine_graph():
    return nx.DiGraph(nx.florentine_families_graph())


"""
Test cases
"""


def test_graph_creation_from_graph(get_florentine_graph):
    graphwrapper = DiGraphWrapper(gr=get_florentine_graph)
    nx.set_edge_attributes(get_florentine_graph, 1, 'type')

    compl_flor = nx.complement(get_florentine_graph)
    get_florentine_graph.add_edges_from(compl_flor.edges(), type=0)

    edge_attr = nx.get_edge_attributes(get_florentine_graph, 'type')

    node_attr = nx.get_node_attributes(graphwrapper.get_graph(), 'type')
    assert edge_attr == node_attr


def test_graph_dim(get_graph_by_dim):
    assert get_graph_by_dim.get_initial_dim() == DIM


def test_edge_attr(get_graph_by_dim, get_random_edge):
    none_edge_count = get_graph_by_dim.get_none_edge_count()
    get_graph_by_dim.set_edge_type(get_random_edge, 1)

    assert get_graph_by_dim.get_edge_type(get_random_edge) == 1
    assert get_graph_by_dim.get_edge_count() == 1
    assert get_graph_by_dim.get_none_edge_count() == none_edge_count - 1
