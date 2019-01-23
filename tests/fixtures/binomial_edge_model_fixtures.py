import pytest
from abc_shadow.model.binomial_edge_graph_model import BinomialEdgeGraphModel


@pytest.fixture
def get_empty_binomial_edge_model(request):
    return BinomialEdgeGraphModel(0)
