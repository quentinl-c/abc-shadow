import numpy as np

import pytest
from abc_shadow.model.two_interactions_graph_model import TwoInteractionsGraphModel


def test_init():
    l1_expected, l2_expected, l12_expected = np.random.uniform(
        0, 100, size=3)
    model = TwoInteractionsGraphModel(l1_expected, l2_expected, l12_expected)

    assert model.l1 == l1_expected
    assert model.l2 == l2_expected
    assert model.l12 == l12_expected
