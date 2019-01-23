import numpy.random as random
from .binomial_edge_graph_model import BinomialEdgeGraphModel


class BinomialGraphModel(BinomialEdgeGraphModel):
    """
    type_value is related to the different edge types
    - 0 : edge doesn't exist (no edge)
    - 1 : edge
    """

    def __init__(self, none_edge_param, edge_param):
        """Initialize Bernouilli (Edge

        Keyword Arguments:
            edge_param {float} -- value of edge parameter
        """
        super().__init__(edge_param)
        self._none_edge_param = none_edge_param

    @property
    def none_edge_param(self):
        """Get edge parameter

        Returns:
            float -- Edge parameter
        """

        return self._none_edge_param

    @none_edge_param.setter
    def none_edge_param(self, new_val):
        """Set _none_edge_param instance variable

        Arguments:
            new_val {int} -- new value
        """
        self._none_edge_param = new_val

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

        super().set_params(args[1])
        self.none_edge_param = args[0]

    def evaluate(self, sample):
        """Given a graph (sample),
        computes the energy function of Edge Model

        Arguments:
            sample {GraphWrapper} -- input sample

        Returns:
            float -- resulting energy
        """

        res = super().evaluate(sample)
        res += self._none_edge_param * sample.get_none_edge_count()
        return res

    def get_local_energy(self, sample, edge, neigh=None):
        """Compute the energy delta regarding
        edge and none edge part.

        Arguments:
            sample {GraphWrapper} -- Input sample
            edge {EdgeId} -- Edge identifier

        Returns:
            float -- Delta energy regarding edge and dyad parts
        """

        res = super().get_local_energy(sample, edge)

        edge_type = sample.get_edge_type(edge)

        if edge_type == 0:
            res += self._none_edge_param

        return res

    def evaluate_from_stats(self, *args):
        """Evaluate the energy (U) from sufficient statistics passed in argument

        Raises:
            ValueError -- If given lentght of stat vector is less than 2

        Returns:
            float -- Energy U
        """

        if len(args) < 2:
            raise ValueError

        res = super().evaluate_from_stats(args[1])
        res += self._none_edge_param * args[0]

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
        dataset["None edges counts"] = [g.get_none_edge_count()
                                        for g in results]
        dataset["Edges counts"] = [g.get_edge_count() for g in results]
        return dataset
