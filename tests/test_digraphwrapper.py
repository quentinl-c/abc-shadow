import networkx as nx
import pytest

from abc_shadow.graph.digraph_wrapper import (DiGraphWrapper,
                                              _lg_directed_in_in,
                                              _lg_directed_out_out)


"""
Test cases
"""


@pytest.mark.usefixtures("get_florentine_digraph")
def test_graph_creation_from_graph(get_florentine_digraph):
    graphwrapper = DiGraphWrapper(gr=get_florentine_digraph)
    nx.set_edge_attributes(get_florentine_digraph, 1, 'type')

    compl_flor = nx.complement(get_florentine_digraph)
    get_florentine_digraph.add_edges_from(compl_flor.edges(), type=0)

    edge_attr_expected = nx.get_edge_attributes(get_florentine_digraph, 'type')

    edge_attr = nx.get_node_attributes(graphwrapper.graph, 'type')
    assert edge_attr == edge_attr_expected

    # Basic Line Grap built upon a in-out neighbourhood
    l_flor = nx.line_graph(get_florentine_digraph)
    nx.set_edge_attributes(l_flor, 'in-out', 'relation')

    # In-in neighbourhood
    in_in_edges = _lg_directed_in_in(get_florentine_digraph).edges()
    l_flor.add_edges_from(in_in_edges, relation='in-in')

    # Out-out neighbourhood
    out_out_edges = _lg_directed_out_out(get_florentine_digraph).edges()
    l_flor.add_edges_from(out_out_edges, relation='out-out')

    l_edge_attr_expected = nx.get_edge_attributes(l_flor, 'relation')
    wrapper_edge_attr = nx.get_edge_attributes(graphwrapper.graph,
                                               'relation')

    assert wrapper_edge_attr == l_edge_attr_expected


@pytest.mark.usefixtures("get_empty_digraph", "get_dig_dim")
def test_graph_dim(get_empty_digraph, get_dig_dim):
    expected_dim = get_dig_dim
    dim = get_empty_digraph.get_initial_dim()

    assert dim == expected_dim


@pytest.mark.usefixtures("get_empty_digraph", "get_random_edge")
def test_edge_attr(get_empty_digraph, get_random_edge):
    none_edge_count_initial = get_empty_digraph.get_none_edge_count()
    get_empty_digraph.set_edge_type(get_random_edge, 1)

    type_expecteed = 1
    actual_type = get_empty_digraph.get_edge_type(get_random_edge)
    assert actual_type == type_expecteed

    edge_count_expected = 1
    edge_count = get_empty_digraph.get_edge_count()
    assert edge_count == edge_count_expected

    none_edge_count_expected = none_edge_count_initial - 1
    none_edge_count = get_empty_digraph.get_none_edge_count()
    assert none_edge_count == none_edge_count_expected
