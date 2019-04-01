# distutils: language=c++
import cython
from libcpp cimport bool, map, pair, vector
from functools import reduce
import networkx as nx


def get_first_set_elmt(s):
    if len(s) > 0:
        return list(s)[0]
    else:
        raise ValueError("set {} is empty".format(s))


def relabel_inv_line_graph(gr):
    renaming_map = dict()

    for n in gr.nodes():
        renaming_map[n] = get_first_set_elmt(
            reduce(lambda x, y: set(x) & set(y), n))

    return nx.relabel_nodes(gr, renaming_map, copy=True)

# cdef map[pair[int, int], vector[pair[int, int]]] from_adjacency_to_cpp_map(dict adj):
#     cdef pair[int, int] key
#     cdef vector[pair[int, int]] neigh
#     cdef map[pair[int, int], vector[pair[int, int]]] res
#     for key, neigh in adj.items():
            
