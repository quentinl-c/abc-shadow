from .dyadic_model import DyadicModel
import numpy.random as random


class MarkovModel(DyadicModel):
    """
    types_values is related to the different edge types
    - 0 : edge doesn't exist (no edge)
    - 1 : in edge
    - 2 : out edge
    - 3 : mutual edge
    """

    type_values = {0, 1, 2, 3}

    def __init__(self, n_edge_p, edge_in_p, edge_out_p, dyad_p, in_p, out_p):
        """Initialize Markovian model

        Arguments:
            edge_in_p {float} -- Edge in parameter
            edge_out_p {float} -- Edge out parameter
            dyad_p {float} -- Dyad parameter
            in_p {float} -- 2-in-star parameter
            out_p {float} -- 2-out-star parameter
        """

        DyadicModel.__init__(self, n_edge_p, edge_in_p, dyad_p)

        self._edge_out_param = edge_out_p
        self._in_param = in_p
        self._out_param = out_p

    def set_edge_out_param(self, new_val):
        """Set _edge_out_param instance variable

        Arguments:
            new_val {int} -- new value
        """

        self._edge_out_param = new_val

    def set_in_param(self, new_val):
        """Set _in_param instance variable

        Arguments:
            new_val {int} -- new value
        """

        self._in_param = new_val

    def set_out_param(self, new_val):
        """Set _out_param instance variable

        Arguments:
            new_val {int} -- new value
        """

        self._out_param = new_val

    def set_params(self, *args):
        """Set all parameter values belonging to the model

        * args corresponds to parameter vector of the model.
        It must be ordered as follows :
        [0] non_edge_param
        [1] edge_in_param
        [2] dyadic_param
        [3] edge_out_param
        [4] in_param
        [5] out_param

        Raises:
            ValueError -- if passed argument is not well sized
        """

        if len(*args) < len(self.__dict__):
            raise ValueError

        super(MarkovModel, self).set_params(args[:3])

        self.set_edge_out_param(args[-3])
        self.set_in_param(args[-2])
        self.set_out_param(args[-1])

    def edge_dyad_delta(self, sample, edge):
        """Compute the energy delta regarding
        edge and dyad part.
        Instead of counting all directed and dyad edges,
        computes only the difference between x_new - x_old

        Arguments:
            sample {GraphWrapper} -- Input sample
            edge {EdgeId} -- Edge identifier

        Returns:
            float -- Delta energy regarding edge and dyad parts
        """

        ego_type = sample.get_edge_type(edge)

        res = 0
        if ego_type == 1:
            res = self._edge_param
        elif ego_type == 2:
            res = self._edge_out_param
        elif ego_type == 3:
            res = self._dyadic_param

        return res

    def evaluate_markov(self, sample, edge):
        """Energy evaluation from the Markov perspective

        Arguments:
            sample {GraphWrapper} -- Input sample
            edge {EdgeId} -- Edge identifier

        Returns:
            float -- energy regarding the Markov part
        """

        ego_type = sample.get_edge_type(edge)

        if ego_type not in {1, 2}:
            return 0

        nbd = sample.get_edge_neighbourhood(edge)

        count = len([e for e in nbd if sample.get_edge_type(e) == ego_type])

        if ego_type == 1:
            return self._in_param * count
        else:
            # This means ego_type is 2
            return self._out_param * count

    def compute_delta(self, mut_sample, edge, new_val):
        """Given a graph sample (mut_sample), an edge on which we will
        affect the new attribute value (new_val),
        computes difference between the new energy (on the modified sample)
        and the previous one (on the initial sample).
        Instead of counting all directed and dyad edges,
        computes only the difference between x_new - x_old

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

        old_energy = self.evaluate_markov(mut_sample, edge)
        old_energy += self.edge_dyad_delta(mut_sample, edge)

        mut_sample.set_edge_type(edge, new_val)

        new_energy = self.evaluate_markov(mut_sample, edge)
        new_energy += self.edge_dyad_delta(mut_sample, edge)

        delta = new_energy - old_energy
        return delta

    def evaluate_from_stats(self, n_edge_count, e_in_count, ed_out_count,
                            d_count, in_count, out_count):
        """Evaluate from given statistics

        Arguments:
            n_edge_count {float} -- None edge count
            e_in_count {float} -- Edge count
            ed_out_count {float} -- [description]
            d_count {float} -- [description]
            in_count {float} -- [description]
            out_count {float} -- [description]

        Returns:
            float -- Energy (U)
        """

        eval = super().evaluate_from_stats(n_edge_count,
                                           e_in_count,
                                           d_count)
        eval += ed_out_count * self._edge_out_param
        eval += in_count * self._in_param
        eval += out_count * self._out_param
        return eval

    @classmethod
    def get_random_candidate_val(cls):
        return random.choice(cls.type_values, 1)[0]

    @staticmethod
    def summary(results):
        """Creates a summary of statistic values according to the current model

        Arguments:
            results {List[GraphWrapper]} -- List of simulation results

        Returns:
            dict -- summary
        """
        dataset = dict()
        dataset['In neigh'] = list()
        dataset['Out neigh'] = list()
        dataset['Edges in'] = list()
        dataset['Edges out'] = list()
        dataset['Dydadics'] = list()
        dataset['None edge'] = list()

        for sample in results:

            in_count = 0
            out_count = 0
            edge_in_count = sample.get_in_edge_count()
            edge_out_count = sample.get_out_edge_count()
            dyad_count = sample.get_dyadic_count()
            none_edge_count = sample.get_none_edge_count()

            for edge in sample.get_elements():
                ego_type = sample.get_edge_type(edge)

                if ego_type not in {1, 2}:
                    continue

                nbd = sample.get_edge_neighbourhood(edge)
                count = len([e for e in nbd
                            if sample.get_edge_type(e) == ego_type])

                if ego_type == 1:
                    in_count += count
                else:
                    # This means ego_type is 2
                    out_count += count

            dataset['In neigh'].append(in_count)
            dataset['Out neigh'].append(out_count)
            dataset['Edges in'].append(edge_in_count)
            dataset['Edges out'].append(edge_out_count)
            dataset['Dydadics'].append(dyad_count)
            dataset['None edge'].append(none_edge_count)
        return dataset
