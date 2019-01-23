"""Markov bridge model module
"""
from .binomial_edge_graph_model import BinomialEdgeGraphModel


class MarkovStarGraphModel(BinomialEdgeGraphModel):
    """
    type_values is related to the different edge types
    - 0 : edge doesn't exist (none edge)
    - 1 : edge exists
    """

    def __init__(self, edge_param, two_star_param):
        """Initialize dyadic model

        Arguments:
            edge_param {float} -- value of edge parameter
            dyadic_param {float} -- value of dyadic parameter
        """
        super().__init__(edge_param)

        self._two_star_param = two_star_param

    def set_params(self, *args):
        """Set all parameter values belonging to the model

        * args corresponds to parameter vector of the model.
        It must be ordered as follows :
        [0] non_edge_param
        [1] edge_param
        [2] bridge_param

        Raises:
            ValueError -- if passed argument is not well sized
        """
        if len(args) < 2:
            raise ValueError

        super().set_params(args[0])
        self.set_two_star_param(args[-1])

    def set_two_star_param(self, new_val):
        """Set _bridge_param instance variable

        Arguments:
            new_val {int} -- new value
        """
        self._two_star_param = new_val

    def get_two_star_param(self):
        """Get dyadic parameter

        Returns:
            float -- Dyadic parameter
        """

        return self._two_star_param

    def evaluate_from_stats(self, *args):

        if len(args) < 2:
            raise ValueError

        edge_side = super().evaluate_from_stats(args[0])
        two_star_side = args[1] * self._two_star_param

        return edge_side + two_star_side

    def evaluate(self, sample):
        """Given a graph (sample),
        computes the energy function of the model

        Arguments:
            sample {GraphWrapper} -- input sample

        Returns:
            float -- resulting energy
        """

        edge_side = super().evaluate(sample)

        two_star_side = self._two_star_param * sample.get_two_star_count()

        return edge_side + two_star_side

    def get_local_energy(self, sample, edge, neigh=None):
        """Compute the local energy.

        Arguments:
            sample {GraphWrapper} -- Input sample
            edge {EdgeId} -- Edge identifier

        Returns:
            float -- Delta energy
        """
        edge_type = sample.get_edge_type(edge)

        res = super().get_local_energy(sample, edge)

        if edge_type == 1:
            res += self._two_star_param * sample.get_local_birdges_count(edge)

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
        return dataset
