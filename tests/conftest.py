import pytest
from .fixtures.graph_fixtures import (get_dim, get_rdn_size, get_empty_graph,
                                      get_random_edge, get_random_edges,
                                      get_one_edge_enabled_graph,
                                      get_florentine_graph,
                                      get_one_star_enabled_graph,
                                      get_random_disct_edge_couple)
from .fixtures.markov_star_model_fixtures import get_empty_markov_star_model
from .fixtures.binomial_edge_model_fixtures import get_empty_binomial_edge_model
from .fixtures.digraph_fixtures import (get_dig_dim, get_empty_digraph,
                                        get_random_diedge,
                                        get_florentine_digraph)
