import pytest
from abc_shadow.model.markov_star_graph_model import MarkovStarGraphModel


@pytest.fixture
def get_empty_markov_star_model(request):
    return MarkovStarGraphModel(0, 0)
