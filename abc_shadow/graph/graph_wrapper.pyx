import networkx as nx
from collections.abc import Iterable
from .utils import relabel_inv_line_graph
from numpy cimport ndarray
import numpy as np
import cython

DEFAULT_DIM = 10
DEFAULT_LABEL = 0


cdef class GraphWrapper(object):

    cdef dict _graph
    cdef dict _vertex

    def __init__(self, dim=DEFAULT_DIM, gr=None, default_label=DEFAULT_LABEL):
        """Initialize Graph Wrapper object
        This is a wrapper over a networkx graph

        The graph model is based on line graph:
        the observed graph is transformed to its
        corresponding line graph.

        Consequently, observed edges are nodes (in line graph)
        and observed nodes bridging two edges are edges (in line graph).

        Keyword Arguments:
            dim {int} -- dimension of initial graph (default: {DEFAULT_DIM})
                         a complete graph dimensionned by dim is initiated
                         if no graph is passed (grap argument is None)
            gr {networkx.Graph} -- input graph (default: {None})
        """
        
        if dim is None:
            self._graph = None
            self._vertex = None

        else:
            if gr is None:
                # Generate a complete graph instead
                intermed_graph = nx.complete_graph(dim)
                graph = nx.line_graph(intermed_graph)
                nx.set_node_attributes(graph, default_label, 'type')

                self._graph = nx.to_dict_of_lists(graph)
                self._vertex = nx.get_node_attributes(graph, 'type')
            else:
                if isinstance(gr, nx.DiGraph) or isinstance(gr, nx.MultiGraph):
                    msg = "⛔️ The graph passed in argument must be a Graph,"\
                        "for wrapping DiGraph, you should use DiGraphWrapper."

                    raise TypeError(msg)

                graph = gr.copy()

                compl_graph = nx.complement(graph)
                nx.set_edge_attributes(graph, 1, 'type')

                graph.add_edges_from(compl_graph.edges(), type=default_label)

                attr = nx.get_edge_attributes(graph, 'type')

                is_key_iterable = all([isinstance(key, Iterable)
                                       for key in attr.keys()])

                if is_key_iterable:
                    attr = {tuple(sorted(key)): val for key,
                            val in attr.items()}

                graph = nx.line_graph(graph)
                nx.set_node_attributes(graph, attr, 'type')

                self._graph = nx.to_dict_of_lists(graph)
                self._vertex = nx.get_node_attributes(graph, 'type')

    def copy(self):
        copy = GraphWrapper(dim=None)
        copy.graph = self.graph
        copy.vertex = self.vertex.copy()
        return copy

    @property
    def vertex(self):
        return self._vertex

    @vertex.setter
    def vertex(self, new_ver):
        self._vertex = new_ver

    @property
    def graph(self):
        """Returns the Networkx graph corresponding to the graph model

        Returns:
            nx.Graph -- Corresponding graph
        """

        return self._graph

    @graph.setter
    def graph(self, new_gr):
        self._graph = new_gr

    def get_initial_graph(self):
        graph = nx.from_dict_of_lists(self.graph)
        inv_line = nx.inverse_line_graph(graph)
        origin = relabel_inv_line_graph(inv_line)

        edges_to_rm = self.get_disabled_edges()
        origin.remove_edges_from(edges_to_rm)
        return origin

    def get_initial_dim(self):
        """Returns the dimension of the initial graph

        Returns:
            int -- Dimension of the initial graph
        """

        graph = nx.from_dict_of_lists(self.graph)
        return len(nx.inverse_line_graph(graph))

    cpdef get_none_edge_count(self):
        """Return the number of nodes labelled as none edge

        Returns:
            int -- number of none 'edges'
        """
        return len(self.get_disabled_edges())

    cpdef get_edge_count(self):
        """Return the number of nodes labelled as directed edge / edge

        Returns:
            int -- number of directed 'edges'
        """

        return len(self.get_enabled_edges())

    cpdef get_elements(self):
        """Get de list of line graph nodes
        -> edges of the initial graph representation

        Returns:
            List[EdgeId] -- list of edge identifiers (tuple)
        """

        return self._vertex.keys()

    cpdef get_edge_type(self, edge_id):
        """Given an edge id
        return its corresponding type

        Arguments:
            edge_id {EdgeId} -- Edge identifier

        Returns:
            int -- Edge type
        """
        return self._vertex[edge_id]

    cpdef is_active_edge(self, edge_id):
        """Returns True if the edge referred by edge_id
        is active (i.e. edge_type != 0)
        False otherwise

        Arguments:
            edge_id {EdgeId} -- Edge identifier

        Returns:
            bool -- True if the edge is active
                    False otherwise
        """
        return self.get_edge_type(edge_id) != 0

    cpdef get_enabled_edges(self):
        return [k for k, e in self._vertex.items() if e != 0]

    cpdef get_disabled_edges(self):
        return [k for k, e in self._vertex.items() if not e]

    cpdef set_edge_type(self, edge_id, new_val):
        """Givent an edge id
        set a new value of its corresponding type

        Arguments:
            edge_id {edgeId} -- Edge identifier
            new_val {int} -- New value
        """

        try:
            val = int(new_val)
        except ValueError:
            msg = "🤯 Edge Type must be an integer"
            raise TypeError(msg)

        self._vertex[edge_id] = val

    cpdef get_edge_neighbourhood(self, edge):
        """Get the neighbourhood of the edge
        All edges connected to 'edge'

        Arguments:
            edge {edgeId} -- Edge identfier

        Returns:
            List[EdgeId] -- Neighbours
        """

        # Returns only active edge in the neighborhood

        return self._graph[edge]

    cpdef get_density(self):
        enabled_edges = len(self.get_enabled_edges())
        disabled_edges = len(self.get_disabled_edges())

        if enabled_edges < 0 and disabled_edges < 0:
            return 0

        d = enabled_edges / (enabled_edges + disabled_edges)
        return d

    cpdef get_edge_type_count(self, t):
        l_edges = [k for k, e in self._vertex.items() if e == t]
        return len(l_edges)

    cpdef get_repulsion_count(self, excluded_labels=None):
        count = 0

        edges = self._vertex.keys()

        for e in edges:
            count += self.get_local_repulsion_count(e,
                                                    excluded_labels=excluded_labels)

        return count / 2

    @cython.boundscheck(False)
    @cython.cdivision(True)
    cpdef ndarray[double] get_interactions_count(self, int inter_nbr):
        cdef ndarray[double] interactions_count = np.zeros(inter_nbr)
        cdef list edges = list(self._vertex.keys())
        cdef ndarray[double] local_count
        for e in edges:
            local_count = self.get_local_interaction_count(e, inter_nbr)
            interactions_count = np.add(interactions_count, local_count)


        return interactions_count / 2
    ##############################################################
    # Local statistics
    ##############################################################

    cpdef get_local_repulsion_count(self, edge, list excluded_labels=None):

        excluded_labels = [] if excluded_labels is None else list(
            excluded_labels)

        cdef int ego_type = self.get_edge_type(edge)

        if ego_type in excluded_labels:
            return 0

        cdef int count = 0
        cdef int label

        for n in self.get_edge_neighbourhood(edge):
            label = self._vertex[n]
            if label != ego_type and label not in excluded_labels:
                count += 1

        return count
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef ndarray[double] get_local_interaction_count(self, tuple edge, int inter_nbr):
        cdef ndarray[double] interactions_count = np.zeros(inter_nbr)
        cdef int ego_type = self._vertex[edge]
        cdef int label, idx

        for n in self._graph[edge]:
            label = self._vertex[n]
            if label != ego_type:
                interactions_count[ego_type + label - 1] += 1

        return interactions_count

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef ndarray[double] get_local_interaction_diff(self, tuple edge, int inter_nbr, int new_ego_type):
        cdef ndarray[double] interactions_count = np.zeros(inter_nbr)
        cdef int current_ego_type = self._vertex[edge]
        cdef int label, idx

        for n in self._graph[edge]:
            label = self._vertex[n]
            if label != current_ego_type:
                interactions_count[current_ego_type + label - 1] -= 1
            if label != new_ego_type:
                interactions_count[new_ego_type + label - 1] += 1
        self._vertex[edge] = new_ego_type
        return interactions_count