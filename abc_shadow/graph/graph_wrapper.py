import networkx as nx
from collections.abc import Iterable
from .utils import relabel_inv_line_graph

DEFAULT_DIM = 10
DEFAULT_LABEL = 0


class GraphWrapper(object):

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

    def get_none_edge_count(self):
        """Return the number of nodes labelled as none edge

        Returns:
            int -- number of none 'edges'
        """
        return len(self.get_disabled_edges())

    def get_edge_count(self):
        """Return the number of nodes labelled as directed edge / edge

        Returns:
            int -- number of directed 'edges'
        """

        return len(self.get_enabled_edges())

    def get_elements(self):
        """Get de list of line graph nodes
        -> edges of the initial graph representation

        Returns:
            List[EdgeId] -- list of edge identifiers (tuple)
        """

        return self.vertex.keys()

    def get_edge_type(self, edge_id):
        """Given an edge id
        return its corresponding type

        Arguments:
            edge_id {EdgeId} -- Edge identifier

        Returns:
            int -- Edge type
        """
        return self.vertex[edge_id]

    def is_active_edge(self, edge_id):
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

    def get_enabled_edges(self):
        return [k for k, e in self.vertex.items() if e != 0]

    def get_disabled_edges(self):
        return [k for k, e in self.vertex.items() if not e]

    def set_edge_type(self, edge_id, new_val):
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

        self.vertex[edge_id] = val

    def get_edge_neighbourhood(self, edge):
        """Get the neighbourhood of the edge
        All edges connected to 'edge'

        Arguments:
            edge {edgeId} -- Edge identfier

        Returns:
            List[EdgeId] -- Neighbours
        """

        # Returns only active edge in the neighborhood

        return self.graph[edge]

    def get_density(self):
        enabled_edges = len(self.get_enabled_edges())
        disabled_edges = len(self.get_disabled_edges())

        if enabled_edges < 0 and disabled_edges < 0:
            return 0

        d = enabled_edges / (enabled_edges + disabled_edges)
        return d

    def get_edge_type_count(self, t):
        l_edges = [k for k, e in self.vertex.items() if e == t]
        return len(l_edges)

    def get_diff_type_count(self):
        count = 0

        enabled_edges = self.get_enabled_edges()

        for e in enabled_edges:
            count += self.get_local_diff_type_count(e)

        return count / 2

    ##############################################################
    # Local statistics
    ##############################################################

    def get_local_diff_type_count(self, edge):
        ego_type = self.get_edge_type(edge)

        if not ego_type:
            return 0

        count = 0

        for n in self.get_edge_neighbourhood(edge):
        # self.get_edge_neighbourhood(edge)
            label = self.vertex[n]
            if label != 0 and label != ego_type:
                count += 1

        return count
