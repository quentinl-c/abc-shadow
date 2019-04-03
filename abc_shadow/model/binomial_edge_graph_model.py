import numpy as np
from .graph_model import GraphModel


class BinomialEdgeGraphModel(GraphModel):
    """
    type_value is related to the different edge types
    - 0 : edge doesn't exist (no edge)
    - 1 : edge

    Binomial Model taking into account just the edge parameter
    """

    type_values = [0, 1]

    def __init__(self, *args):
        """Initialize Bernouilli (Edge

        Keyword Arguments:
            edge_param {float} -- value of edge parameter
        """
        if len(args) != 1:
            raise ValueError
        super().__init__(*args)

    @property
    def edge_param(self):
        """Get edge parameter

        Returns:
            float -- Edge parameter
        """

        return self._params[0]

    def set_params(self, *args):
        """Set all parameter values belonging to the model

        * args corresponds to parameter vector of the model.
        It must be ordered as follows :
        [0] non_edge_param
        [1] edge_param

        Raises:
            ValueError -- if passed argument is not well sized
        """
        if len(args) != 1:
            raise ValueError

        super().set_params(*args)

    def evaluate(self, sample):
        """Given a graph (sample),
        computes the energy function of Edge Model

        Arguments:
            sample {GraphWrapper} -- input sample

        Returns:
            float -- resulting energy
        """

        return self.edge_param * sample.get_edge_count()

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

        res = 0

        if edge_type == 1:
            res = self.edge_param

        return res

    def evaluate_from_stats(self, *args):
        """Evaluate the energy (U) from sufficient statistics passed in argument

        Raises:
            ValueError -- If given lentght of stat vector is less than 2

        Returns:
            float -- Energy U
        """

        if len(args) != 1:
            raise ValueError

        return super().evaluate_from_stats(*args)

    @staticmethod
    def get_delta_stats(mut_sample, edge, new_label):
        res = np.zeros(1)
        old_label = mut_sample.get_edge_type(edge)
        mut_sample.set_edge_type(edge, new_label)
        res[0] = old_label - new_label
        return res

    @staticmethod
    def summary_dict(results):
        """Creates a summary of configuration values according to the current model

        Arguments:
            results {List[GraphWrapper]} -- List of simulation results

        Returns:
            dict -- summary
        """

        dataset = dict()
        dataset['Edges counts'] = [g.get_edge_count() for g in results]
        return dataset
