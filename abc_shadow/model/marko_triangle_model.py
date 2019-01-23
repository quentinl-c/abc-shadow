"""Markov bridge model module
"""
from .markov_star_graph_model import MarkovStarGraphModel


class MarkovTriangleGraphModel(MarkovStarGraphModel):
    """
    type_values is related to the different edge types
    - 0 : edge doesn't exist (none edge)
    - 1 : edge exists
    """

    def __init__(self, edge_param, two_star_param, triangle_param):
        """Initialize dyadic model

        Arguments:
            edge_param {float} -- value of edge parameter
            dyadic_param {float} -- value of dyadic parameter
        """
        super().__init__(edge_param, two_star_param)
        self._triangle_param = triangle_param

    def set_params(self, *args):
        """Set all parameter values belonging to the model

        * args corresponds to parameter vector of the model.
        It must be ordered as follows :
        [0] non_edge_param
        [1] edge_param

        Raises:
            ValueError -- if passed argument is not well sized
        """
        if len(args) < 2:
            raise ValueError

        super().set_params(*args[:2])
        self.set_triangle_param(args[1])

    def set_triangle_param(self, new_val):
        """Set _triangle_param instance variable

        Arguments:
            new_val {int} -- new value
        """

        self._triangle_param = new_val

    def get_triangle_param(self):
        """Get triangle parameter

        Returns:
            float -- Dyadic parameter
        """

        return self._triangle_param

    def evaluate_from_stats(self, *args):

        if len(args) < 2:
            raise ValueError

        edge_star_side = super().evaluate_from_stats(*args[:-1])
        triangle_side = args[-1] * self._triangle_param

        return edge_star_side + triangle_side

    def evaluate(self, sample):
        """Given a graph (sample),
        computes the energy function of Dyadic Model

        Arguments:
            sample {GraphWrapper} -- input sample

        Returns:
            float -- resulting energy
        """

        edge_star_side = super().evaluate(sample)

        triangle_side = self._triangle_param * sample.get_triangle_count()

        return edge_star_side + triangle_side

    def get_local_energy(self, sample, edge, neigh=None):
        """Compute the energy delta regarding
        edge and dyad part.

        Arguments:
            sample {GraphWrapper} -- Input sample
            edge {EdgeId} -- Edge identifier

        Returns:
            float -- Delta energy regarding edge and dyad parts
        """
        res = super().get_local_energy(sample, edge)

        edge_type = sample.get_edge_type(edge)

        if edge_type == 1:
            res += self._triangle_param * \
                sample.get_local_triangles_count(edge)
        return res

    @staticmethod
    def summary(results):
        """Creates a summary of configuration values according to the current model

        Arguments:
            results {List[GraphWrapper]} -- List of simulation results

        Returns:
            dict -- summary
        """
        dataset = dict()

        dataset['Edges counts'] = [g.get_edge_count() for g in results]
        dataset['2 star counts'] = [g.get_two_star_count() for g in results]
        dataset['Triangle counts'] = [g.get_triangles_count() for g in results]
        return dataset
