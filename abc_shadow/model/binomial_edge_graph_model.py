import numpy.random as random
from .graph_model import GraphModel


class BinomialEdgeGraphModel(GraphModel):
    """
    type_value is related to the different edge types
    - 0 : edge doesn't exist (no edge)
    - 1 : edge

    Binomial Model taking into account just the edge parameter
    """

    type_values = {0, 1}

    def __init__(self, edge_param):
        """Initialize Bernouilli (Edge

        Keyword Arguments:
            edge_param {float} -- value of edge parameter
        """
        self._edge_param = edge_param

    @property
    def edge_param(self):
        """Get edge parameter

        Returns:
            float -- Edge parameter
        """

        return self._edge_param

    @edge_param.setter
    def edge_param(self, new_val):
        """Set _edge_param instance variable

        Arguments:
            new_val {int} -- new value
        """
        self._edge_param = new_val

    def set_params(self, *args):
        """Set all parameter values belonging to the model

        * args corresponds to parameter vector of the model.
        It must be ordered as follows :
        [0] non_edge_param
        [1] edge_param

        Raises:
            ValueError -- if passed argument is not well sized
        """
        if len(args) < 1:
            raise ValueError

        self.edge_param = args[0]

    def evaluate(self, sample):
        """Given a graph (sample),
        computes the energy function of Edge Model

        Arguments:
            sample {GraphWrapper} -- input sample

        Returns:
            float -- resulting energy
        """

        return self._edge_param * sample.get_edge_count()

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
            res = self._edge_param

        return res

    def evaluate_from_stats(self, *args):
        """Evaluate the energy (U) from sufficient statistics passed in argument

        Raises:
            ValueError -- If given lentght of stat vector is less than 2

        Returns:
            float -- Energy U
        """

        if len(args) < 1:
            raise ValueError

        edge_side = self._edge_param * args[0]

        return edge_side

    @classmethod
    def get_random_candidate_val(cls, p=None):
        return random.choice(list(cls.type_values), p=p)

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
        return dataset
