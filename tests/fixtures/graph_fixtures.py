import pytest
import numpy as np
import networkx as nx
from abc_shadow.graph.graph_wrapper import GraphWrapper

DIM = 16
RDN_NBR = 5


@pytest.fixture
def get_dim(request):
    return DIM


@pytest.fixture
def get_rdn_size(request):
    return RDN_NBR


@pytest.fixture
def get_empty_graph(request, get_dim):
    graph_wrapper = GraphWrapper(get_dim)
    return graph_wrapper


@pytest.fixture
def get_random_edge(request, get_empty_graph):
    edges = list(get_empty_graph.get_elements())
    random_edge = edges[np.random.choice(len(edges))]
    return random_edge


@pytest.fixture
def get_random_edges(request, get_empty_graph, get_rdn_size):
    edges = list(get_empty_graph.get_elements())
    random_indices = np.random.choice(
        len(edges), replace=False, size=get_rdn_size)
    return [edges[i] for i in random_indices]


@pytest.fixture
def get_florentine_graph(request):
    return nx.florentine_families_graph()


@pytest.fixture
def get_one_edge_enabled_graph(request, get_dim):
    g = nx.Graph()
    g.add_nodes_from(range(get_dim))
    i, j = sorted(np.random.choice(range(get_dim), replace=False, size=2))
    g.add_edge(i, j)
    gr = GraphWrapper(gr=g)

    return gr


@pytest.fixture
def get_one_star_enabled_graph(request, get_dim):
    g = nx.Graph()
    g.add_nodes_from(range(get_dim))
    i, j, k = sorted(np.random.choice(range(get_dim), replace=False, size=3))
    g.add_edge(i, j)
    g.add_edge(j, k)
    gr = GraphWrapper(gr=g)

    return gr


@pytest.fixture
def get_random_disct_edge_couple(request, get_empty_graph, get_random_edge):
    ego1 = get_random_edge
    print(ego1)
    ego1_neigh = list(get_empty_graph.graph().neighbors(ego1))
    all_neighs = [get_empty_graph.graph().neighbors(n)
                  for n in ego1_neigh]
    all_neighs.extend(ego1_neigh)

    all_edges = list(get_empty_graph.get_elements())
    available_edges = list(set(all_edges) - set(all_neighs))
    print(available_edges)
    ego2 = available_edges[np.random.choice(len(available_edges))]
    return ego1, ego2
