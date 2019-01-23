import random

import networkx as nx
import pytest

from abc_shadow.graph.digraph_wrapper import DiGraphWrapper

DIM = 16


@pytest.fixture
def get_dig_dim(request):
    return DIM


@pytest.fixture
def get_empty_digraph(request, get_dig_dim):
    graph_wrapper = DiGraphWrapper(get_dig_dim)
    return graph_wrapper


@pytest.fixture
def get_random_diedge(get_empty_digraph):
    edges = get_empty_digraph.get_elements()
    random_edge = random.sample(list(edges), k=1)[0]
    return random_edge


@pytest.fixture
def get_florentine_digraph():
    return nx.DiGraph(nx.florentine_families_graph())
