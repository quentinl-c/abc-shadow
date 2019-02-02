import pytest
import networkx as nx
import numpy.random as random
from abc_shadow.graph.graph_wrapper import GraphWrapper


"""
Test cases
"""


@pytest.mark.usefixtures("get_florentine_graph")
def test_graph_creation_from_graph(get_florentine_graph):
    graphwrapper = GraphWrapper(get_florentine_graph)
    nx.set_edge_attributes(get_florentine_graph, 1, 'type')

    compl_flor = nx.complement(get_florentine_graph)
    get_florentine_graph.add_edges_from(compl_flor.edges(), type=0)

    edge_attr_expected = set({tuple(sorted(key)): int(val)
                              for key, val in nx.get_edge_attributes(
        get_florentine_graph, 'type').items()})

    edge_attr = set(graphwrapper.vertex)
    assert edge_attr == edge_attr_expected


@pytest.mark.usefixtures("get_empty_graph", "get_dim")
def test_graph_dim(get_empty_graph, get_dim):
    dim_expected = get_dim
    dim = get_empty_graph.get_initial_dim()
    assert dim == dim_expected


@pytest.mark.usefixtures("get_empty_graph",)
def test_edge_attr(get_empty_graph, get_random_edge):
    e = get_random_edge
    none_edge_count_initial = get_empty_graph.get_none_edge_count()
    get_empty_graph.set_edge_type(e, 1)

    status_expected = 1
    status = get_empty_graph.get_edge_type(e)
    assert status == status_expected

    edge_count_expected = 1
    edge_count = get_empty_graph.get_edge_count()
    assert edge_count == edge_count_expected

    none_edge_count_expected = none_edge_count_initial - 1
    none_edge_count = get_empty_graph.get_none_edge_count()
    assert none_edge_count == none_edge_count_expected


@pytest.mark.usefixtures("get_empty_graph", "get_random_edge")
def test_is_enabled_edge(get_empty_graph, get_random_edge):
    e = get_random_edge
    get_empty_graph.set_edge_type(e, 0)
    disabled_status = get_empty_graph.is_active_edge(e)
    assert not disabled_status

    get_empty_graph.set_edge_type(e, 1)
    enabled_status = get_empty_graph.is_active_edge(e)
    assert enabled_status


@pytest.mark.usefixtures("get_empty_graph", "get_random_edges")
def test_get_enabled_edges(get_empty_graph, get_random_edges):
    edges_expected = sorted(get_random_edges)
    for e in edges_expected:
        get_empty_graph.set_edge_type(e, 1)

    enabled_edges = sorted(get_empty_graph.get_enabled_edges())

    assert enabled_edges == edges_expected


@pytest.mark.usefixtures("get_empty_graph", "get_random_edges")
def test_get_disabled_edgesdisabled(get_empty_graph, get_random_edges):
    gr = get_empty_graph

    disabled_edges_before = sorted(gr.get_disabled_edges())
    disabled_edges_before_expected = sorted(gr.get_elements())
    assert disabled_edges_before == disabled_edges_before_expected

    enabled_edges = sorted(get_random_edges)
    for e in enabled_edges:
        get_empty_graph.set_edge_type(e, 1)

    disabled_after_edges_expected = sorted(list(
        set(disabled_edges_before) - set(enabled_edges)))
    disabled_edges_after = sorted(gr.get_disabled_edges())

    assert disabled_after_edges_expected == disabled_edges_after


@pytest.mark.usefixtueres("get_empty_graph", "get_random_edge")
def test_copy(get_empty_graph, get_random_edges):
    gr = get_empty_graph

    for e in get_random_edges:
        gr.set_edge_type(e, 1)

    copy_gr = gr.copy()

    expected = 1
    for e in get_random_edges:
        assert copy_gr.get_edge_type(e) == expected

    switched_edge = get_random_edges[0]
    gr.set_edge_type(switched_edge, 0)

    assert copy_gr.get_edge_type(switched_edge) == expected


@pytest.mark.usefixtueres("get_empty_graph", "get_random_edge")
def test_get_local_diff_type_count(get_empty_graph, get_random_edge):
    gr = get_empty_graph
    ego = get_random_edge

    neigh = list(gr.get_edge_neighbourhood(ego))

    ego_type = 1
    gr.set_edge_type(ego, ego_type)

    designated_survivors = random.choice(len(neigh), replace=False, size=5)
    boundary = random.randint(5)
    alters = [neigh[i] for i in designated_survivors[:boundary]]
    others = [neigh[i] for i in designated_survivors[boundary:]]

    for a in alters:
        gr.set_edge_type(a, 2)

    for o in others:
        gr.set_edge_type(o, 1)

    expected = len(alters)
    local_diff_type_count = gr.get_local_diff_type_count(ego)

    assert local_diff_type_count == expected


@pytest.mark.usefixtueres("get_empty_graph")
def test_get_diff_type_count(get_empty_graph):
    gr = get_empty_graph
    ego1 = (1, 2)
    gr.set_edge_type(ego1, 1)

    ego2 = (8, 9)
    gr.set_edge_type(ego2, 1)

    alters1 = [(1, 3), (2, 4)]
    others1 = [(1, 5), (2, 6)]

    alters2 = [(7, 8), (9, 10)]
    others2 = [(7, 11), (7, 12)]

    for a1, a2 in zip(alters1, alters2):
        gr.set_edge_type(a1, 2)
        gr.set_edge_type(a2, 2)

    for o1, o2 in zip(others1, others2):
        gr.set_edge_type(o1, 1)
        gr.set_edge_type(o2, 1)

    expected = len(alters1) + len(alters2) + len(others1) + len(others2)
    diff_type_count = gr.get_diff_type_count()
    assert diff_type_count == expected
