import networkx as nx
from collections.abc import Iterable
from .utils import get_first_set_elmt, relabel_inv_line_graph

DEFAULT_DIM = 10


class GraphWrapper(object):

    def __init__(self, dim=DEFAULT_DIM, gr=None):
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

        else:
            if gr is None:
                # Generate a complete graph instead
                graph = nx.complete_graph(dim)

                self._graph = nx.line_graph(graph)
                nx.set_node_attributes(self._graph, 0, 'type')

            else:
                if isinstance(gr, nx.DiGraph) or isinstance(gr, nx.MultiGraph):
                    msg = "⛔️ The graph passed in argument must be a Graph,"\
                        "for wrapping DiGraph, you should use DiGraphWrapper."

                    raise TypeError(msg)

                graph = gr.copy()

                compl_graph = nx.complement(graph)
                nx.set_edge_attributes(graph, 1, 'type')

                graph.add_edges_from(compl_graph.edges(), type=0)

                attr = nx.get_edge_attributes(graph, 'type')

                is_key_iterable = all([isinstance(key, Iterable)
                                       for key in attr.keys()])

                if is_key_iterable:
                    attr = {tuple(sorted(key)): val for key,
                            val in attr.items()}

                graph = nx.line_graph(graph)
                nx.set_node_attributes(graph, attr, 'type')

                self._graph = graph

    def copy(self):
        copy = GraphWrapper(dim=None)
        copy._set_graph(self._graph.copy())
        return copy

    def _set_graph(self, new_gr):
        self._graph = new_gr

    @property
    def graph(self):
        """Returns the Networkx graph corresponding to the graph model

        Returns:
            nx.Graph -- Corresponding graph
        """

        return self._graph

    def get_initial_graph(self):
        inv_line = nx.inverse_line_graph(self.graph)
        origin = relabel_inv_line_graph(inv_line)

        edges_to_rm = self.get_disabled_edges()
        origin.remove_edges_from(edges_to_rm)
        return origin

    def get_initial_dim(self):
        """Returns the dimension of the initial graph

        Returns:
            int -- Dimension of the initial graph
        """

        return len(nx.inverse_line_graph(self._graph))

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

        return self._graph.nodes()

    def get_particle(self, id):
        """Wrapper of get_edge_type method

        Arguments:
            id {EdgeId} -- Edge identifier

        Returns:
            int -- Edge type
        """

        return self.get_edge_type(id)

    def get_edge_type(self, edge_id):
        """Given an edge id
        return its corresponding type

        Arguments:
            edge_id {EdgeId} -- Edge identifier

        Returns:
            int -- Edge type
        """
        return self._graph.nodes[edge_id]['type']

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
        return [e for e in self.get_elements() if self.is_active_edge(e)]

    def get_enabled_graph(self):
        return nx.subgraph(self._graph, nbunch=self.get_enabled_edges())

    def get_disabled_edges(self):
        return [e for e in self.get_elements() if not self.is_active_edge(e)]

    def set_particle(self, id, new_val):
        """Wrapper of set_edge_type

        Arguments:
            id {edgeId} -- Edge identifier
            new_val int} -- New value
        """

        self.set_edge_type(id, new_val)

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

        self._graph.nodes[edge_id]['type'] = val

    def get_edge_neighbourhood(self, edge):
        """Get the neighbourhood of the edge
        All enabled edges connected to 'edge'

        Arguments:
            edge {edgeId} -- Edge identfier

        Returns:
            List[EdgeId] -- Neighbours
        """

        # Returns only active edge in the neighborhood
        neighs = [t for t in self._graph.neighbors(
            edge) if self.is_active_edge(t)]

        return neighs

    def get_density(self):
        enabled_edges = len(self.get_enabled_edges())
        disabled_edges = len(self.get_disabled_edges())

        if enabled_edges < 0 and disabled_edges < 0:
            return 0

        d = enabled_edges / (enabled_edges + disabled_edges)
        return d

    def get_two_star_count(self):
        enabled_graph = self.get_enabled_graph()
        two_star_count = len(enabled_graph.edges())
        return two_star_count

    def get_edge_type_count(self, t):
        l_edges = [e for e in self.graph.nodes(data='type') if e[1] == t]
        return len(l_edges)

    def get_diff_type_count(self):
        count = 0

        enabled_graph = self.get_enabled_edges()
        labels = nx.get_node_attributes(self.graph, 'type')

        for e in enabled_graph:

            ego_type = labels[e]
            neigh_labels = [labels[n] for n in self.graph.neighbors(e)]

            for t in neigh_labels:
                if t != 0 and t != ego_type:
                    count += 1

        # Less efficient
        # enabled_graph = self.get_enabled_edges()
        # count = 0

        # for e in enabled_graph:
        #     count += self.get_local_diff_type_count(e)

        return count / 2

    def get_heffect_count(self):
        pass
    ##############################################################
    # Local statistics
    ##############################################################

    def get_local_heffect(self, edge):
        self._graph.neighbors(edge)

    def get_local_diff_type_count(self, edge):
        ego_type = self.get_edge_type(edge)

        if not ego_type:
            return 0

        neighs = self._graph.neighbors(edge)
        # self.get_edge_neighbourhood(edge)
        count = 0
        neigh_labels = [self.get_edge_type(n) for n in neighs]
        for t in neigh_labels:
            if t != 0 and t != ego_type:
                count += 1

        return count

    def get_local_birdges_count(self, edge):
        neighs = self.get_edge_neighbourhood(edge)
        return len(neighs)

    def get_local_triangles_count(self, edge):
        neighs = self.get_edge_neighbourhood(edge)

        is_tuples_list = all(
            [isinstance(n, Iterable) and len(n) == 2 for n in neighs])

        if not is_tuples_list:
            msg = "🤯 Edges must be formatted as follows : (node1, node2)"
            raise TypeError(msg)

        nodes = [get_first_set_elmt(set(n) - set(edge)) for n in neighs]
        nodes_set = set(nodes)

        return len(nodes) - len(nodes_set)

    def get_triangles_count(self):
        initial_graph = self.get_initial_graph()

        # For each node, retrieve local triangles count -> list
        triangles = nx.triangles(initial_graph).values()

        triangles_count = sum(triangles) / 3
        return triangles_count
