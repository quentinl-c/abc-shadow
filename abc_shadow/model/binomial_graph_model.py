import numpy.random as random
from .graph_model import GraphModel


class BinomialGraphModel(GraphModel):
    """
    type_value is related to the different edge types
    - 0 : edge doesn't exist (no edge)
    - 1 : edge
    """

    type_values = {0, 1}

    def __init__(self, none_edge_param, edge_param):
        """Initialize Bernouilli (Edge

        Keyword Arguments:
            edge_param {float} -- value of edge parameter
        """
        self._none_edge_param = none_edge_param
        self._edge_param = edge_param

    def set_none_edge_param(self, new_val):
        """Set _none_edge_param instance variable

        Arguments:
            new_val {int} -- new value
        """
        self._none_edge_param = new_val

    def set_edge_param(self, new_val):
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
        if len(args) < 2:
            raise ValueError

        self.set_none_edge_param(args[0])
        self.set_edge_param(args[1])

    def get_none_edge_param(self):
        """Get edge parameter

        Returns:
            float -- Edge parameter
        """

        return self._none_edge_param

    def get_edge_param(self):
        """Get edge parameter

        Returns:
            float -- Edge parameter
        """

        return self._edge_param

    def evaluate(self, sample):
        """Given a graph (sample),
        computes the energy function of Edge Model

        Arguments:
            sample {GraphWrapper} -- input sample

        Returns:
            float -- resulting energy
        """

        return (self._none_edge_param * sample.get_none_edge_count() +
                self._edge_param * sample.get_edge_count())

    def edge_delta(self, sample, edge):
        """Compute the energy delta regarding
        edge and none edge part.

        Arguments:
            sample {GraphWrapper} -- Input sample
            edge {EdgeId} -- Edge identifier

        Returns:
            float -- Delta energy regarding edge and dyad parts
        """
        ego_type = sample.get_edge_type(edge)

        res = 0
        if ego_type == 0:
            res = self._none_edge_param
        if ego_type == 1:
            res = self._edge_param

        return res

    def compute_delta(self, mut_sample, edge, new_val):
        """Given a graph sample (mut_sample), an edge on which we will
        affect the new attribute value (new_val),
        computes difference between the new energy (on the modified sample)
        and the previous one (on the initial sample).
        Instead of counting all directe edges,
        computes only the difference between x_new - x_old

        Arguments:
            mut_sample {GraphWrapper} -- initial sample
                                         (mutable - reference passing)
                                         by side effect, one will be modified
            edge {tuple(int,int)} -- designated edge
                                     (for which the attribute will be modified)
            new_val {int} -- new attribute value comprise

        Returns:
            float -- Energy delta between modified sample and initial one
        """

        old_energy = self.edge_delta(mut_sample, edge)

        mut_sample.set_edge_type(edge, new_val)

        # Computes the delta between old and new energy
        new_energy = self.edge_delta(mut_sample, edge)
        delta = new_energy - old_energy

        return delta

    def evaluate_from_stats(self, *args):
        """Evaluate the energy (U) from sufficient statistics passed in argument

        Raises:
            ValueError -- If given lentght of stat vector is less than 2

        Returns:
            float -- Energy U
        """

        if len(args) < 2:
            raise ValueError

        edge_side = self._none_edge_param * args[0]
        edge_side += self._edge_param * args[1]

        return edge_side

    @classmethod
    def get_random_candidate_val(cls):
        return random.choice(list(cls.type_values), 1)[0]

    @staticmethod
    def summary(results):
        """Creates a summary of configuration values according to the current model

        Arguments:
            results {List[GraphWrapper]} -- List of simulation results

        Returns:
            dict -- summary
        """

        dataset = dict()
        dataset['None edges counts'] = [g.get_none_edge_count()
                                        for g in results]
        dataset['Edges counts'] = [g.get_edge_count() for g in results]
        return dataset
