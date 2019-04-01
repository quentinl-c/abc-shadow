import numpy.random as random
from .graph_model import GraphModel
from collections import Iterable

class BinomialGraphModel(GraphModel):
    """
    type_value is related to the different edge types
    - 0 : edge doesn't exist (no edge)
    - 1 : edge
    """

    def __init__(self, *args):
        """Initialize Bernouilli (Edge

        Keyword Arguments:
            edge_param {float} -- value of edge parameter
        """
        if len(args) != 2:
            raise ValueError

        super().__init__(*args)

    @property
    def none_edge_param(self):
        """Get none edge parameter

        Returns:
            float -- Edge parameter
        """

        return self._params[0]

    @property
    def edge_param(self):
        """Get edge parameter

        Returns:
            float -- Edge parameter
        """

        return self._params[1]

    def set_params(self, *args):
        """Set all parameter values belonging to the model

        * args corresponds to parameter vector of the model.
        It must be ordered as follows :
        [0] non_edge_param
        [1] edge_param

        Raises:
            ValueError -- if passed argument is not well sized
        """
        if len(args) != 2:
            raise ValueError

        return super().set_params(*args)

    def evaluate(self, sample):
        """Given a graph (sample),
        computes the energy function of Edge Model

        Arguments:
            sample {GraphWrapper} -- input sample

        Returns:
            float -- resulting energy
        """
        res = self.none_edge_param * sample.get_none_edge_count()
        res = self.edge_param * sample.get_edge_count()
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

        edge_type = sample.get_edge_type(edge)

        if edge_type == 0:
            return self.none_edge_param
        else:
            return self.edge_param

    def evaluate_from_stats(self, *args):
        """Evaluate the energy (U) from sufficient statistics passed in argument

        Raises:
            ValueError -- If given lentght of stat vector is less than 2

        Returns:
            float -- Energy U
        """

        if len(args) != 2:
            raise ValueError

        return super().evaluate_from_stats(*args)

    @staticmethod
    def summary(results):
        """Creates a summary of configuration values according to the current model

        Arguments:
            results {List[GraphWrapper]} -- List of simulation results

        Returns:
            dict -- summary
        """
        if not isinstance(results, Iterable):
            results = [results]
        data = dict()
        data["None-edges-counts"] = [g.get_none_edge_count()
                                        for g in results]
        data["Edges-counts"] = [g.get_edge_count() for g in results]
        return data
