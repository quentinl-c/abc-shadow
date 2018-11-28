from networkx.generators.line import _edge_func
from .graph_wrapper import GraphWrapper
import networkx as nx

DEFAULT_DIM = 10


class DiGraphWrapper(GraphWrapper):

    def __init__(self, dim=DEFAULT_DIM, gr=None):

        super().__init__(dim=None)

        if gr is None:
            # Generate a complete graph instead
            graph = nx.complete_graph(dim, nx.DiGraph())
            nx.set_edge_attributes(graph, 0, 'type')
        else:
            if not isinstance(gr, nx.DiGraph):
                msg = "⛔️ The graph passed in argument must be a DiGraph,"\
                      "for wrapping simple Graph, you should use GraphWrapper."

                raise TypeError(msg)

            graph = gr.copy()
            nx.set_edge_attributes(graph, 1, 'type')

            compl_graph = nx.complement(graph)
            graph.add_edges_from(compl_graph.edges(), type=0)

        # Initial dimension of the input graph (ie. number of nodes)
        self.__dim = len(graph.nodes())

        self._graph = directed_line_graph(graph)

    def get_initial_dim(self):
        """Returns the dimension of the initial graph

        Returns:
            int -- Dimension of the initial graph
        """

        return self.__dim

    def get_in_out_count(self):
        """Returns the number of in - out relations in the line graph representation

        Returns:
            int -- number of in - out (edge) relation
        """

        bridges = 0
        active_edges = self.get_enabled_edges()
        edge_attr = nx.get_edge_attributes(self._graph, 'relation').items()
        for e, r_attr in edge_attr:
            if(r_attr == 'in-out' and e[0] in active_edges and
               e[1] in active_edges):
                bridges += 1
        return bridges

"""
====================
Additional functions
====================
"""


def directed_line_graph(gr):
    attr = nx.get_edge_attributes(gr, 'type')

    # Draw the Line Graph of gr
    # Basic Line Grap built upon a in-out neighbourhood
    l_gr = nx.line_graph(gr)
    nx.set_edge_attributes(l_gr, 'in-out', 'relation')

    # In-in neighbourhood
    in_in_edges = _lg_directed_in_in(gr).edges()
    l_gr.add_edges_from(in_in_edges, relation='in-in')

    # Out-out neighbourhood
    out_out_edges = _lg_directed_out_out(gr).edges()
    l_gr.add_edges_from(out_out_edges, relation='out-out')

    nx.set_node_attributes(l_gr, attr, 'type')

    return l_gr

"""
Modified built-in functions (nx.generators.line) for Line Graph generation
(for DiGraph)
Includes :
* in-in neighbourhood
* out-out neighbourhood
"""


def _lg_directed_in_in(G, create_using=None):
    """Return the line graph L of the (multi)digraph G.

    Edges in G appear as nodes in L, represented as tuples of the form (u,v)
    or (u,v,key) if G is a multidigraph. A node in L corresponding to the edge
    (u,v) is connected to every node corresponding to an edge (v,w).

    Parameters
    ----------
    G : digraph
        A directed graph or directed multigraph.
    create_using : None
        A digraph instance used to populate the line graph
        (according to in-in neighbourhood).

    """
    L = nx.empty_graph(0, create_using, default=G.__class__)

    # Create a graph specific edge function.
    get_edges = _edge_func(G)

    for from_node in get_edges():
        # from_node is: (u,v) or (u,v,key)
        L.add_node(from_node)
        for to_node in G.in_edges(from_node[1]):
            if from_node != to_node:
                L.add_edge(from_node, to_node)

    return L


def _lg_directed_out_out(G, create_using=None):
    """Return the line graph L of the (multi)digraph G.

    Edges in G appear as nodes in L, represented as tuples of the form (u,v)
    or (u,v,key) if G is a multidigraph. A node in L corresponding to the edge
    (u,v) is connected to every node corresponding to an edge (v,w).

    Parameters
    ----------
    G : digraph
        A directed graph or directed multigraph.
    create_using : None
        A digraph instance used to populate the line graph
        (according to out-out neighbourhood).

    """
    L = nx.empty_graph(0, create_using, default=G.__class__)

    # Create a graph specific edge function.
    get_edges = _edge_func(G)

    for from_node in get_edges():
        # from_node is: (u,v) or (u,v,key)
        L.add_node(from_node)
        for to_node in get_edges(from_node[0]):
            if from_node != to_node:
                L.add_edge(from_node, to_node)
    return L
