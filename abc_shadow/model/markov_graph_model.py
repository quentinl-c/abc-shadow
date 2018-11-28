"""Markov bridge model module
"""
import numpy.random as random
from .binomial_edge_graph_model import BinomialEdgeGraphModel

D_DYAD_PARAM = 1.0


class MarkovGraphModel(BinomialEdgeGraphModel):
    """
    type_values is related to the different edge types
    - 0 : edge doesn't exist (none edge)
    - 1 : edge exists
    """

    type_values = {0, 1}

    def __init__(self, edge_param, two_star_param, triangle_param):
        """Initialize dyadic model

        Arguments:
            edge_param {float} -- value of edge parameter
            dyadic_param {float} -- value of dyadic parameter
        """
        super().__init__(edge_param)

        self._two_star_param = two_star_param
        self._triangle_param = triangle_param

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

        super().set_params(*args[:1])
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

        edge_side = super().evaluate_from_stats(*args[0:1])
        bridge_side = args[-1] * self._two_star_param

        return edge_side + bridge_side

    def evaluate(self, sample):
        """Given a graph (sample),
        computes the energy function of Dyadic Model

        Arguments:
            sample {GraphWrapper} -- input sample

        Returns:
            float -- resulting energy
        """

        edge_side = super().evaluate(sample)

        bridge_side = self._two_star_param * sample.get_dyadic_count()

        return edge_side + bridge_side

    def get_local_energy(self, sample, edge):
        """Compute the energy delta regarding
        edge and dyad part.

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
            res += self._two_star_param * sample.get_local_birdges_count(edge)
            res += self._triangle_param * \
                sample.get_local_triangles_count(edge)
        return res

    def compute_delta(self, mut_sample, edge, new_val):
        """Given a graph sample (mut_sample), an edge on which we will
        affect the new attribute value (new_val),
        computes difference between the new energy (on the modified sample)
        and the previous one (on the initial sample).
        Instead of counting all directed and dyad edges,
        computes only the difference between x_new - x_old.

        Arguments:
            mut_sample {GraphWrapper} -- initial sample
                                         (mutable - reference passing)
                                         by side effect, one will be modified
            edge {EdgeId} -- designated edge
                                     (for which the attribute will be modified)
            new_val {int} -- new attribute value
        Returns:
            float -- Energy delta between modified sample and initial one
        """

        old_energy = self.get_local_energy(mut_sample, edge)

        mut_sample.set_edge_type(edge, new_val)

        # Computes the delta between old and new energy
        new_energy = self.get_local_energy(mut_sample, edge)

        delta = new_energy - old_energy
        return delta

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
        dataset['2 star counts'] = [g.get_bridges_count() for g in results]
        dataset['Triangle counts'] = [g.get_triangles_count() for g in results]
        return dataset
