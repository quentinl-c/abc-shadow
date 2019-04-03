import pytest
from abc_shadow.model.potts_graph_model import PottsGraphModel


@pytest.mark.usefixture("get_empty_graph", "get_random_edge")
def test_compute_delta_stats(get_empty_graph, get_random_edge):
    sample = get_empty_graph
    e = get_random_edge
    model = PottsGraphModel(0, 0, 0)
    interactions_before = sample.get_local_interaction_count(e, 3)
    delta_stats = model.get_delta_stats(sample, e, 1)
    interactions_after = sample.get_local_interaction_count(e, 3)
    expected_delta_stats = interactions_after - interactions_before
    assert delta_stats.all() == expected_delta_stats.all()
