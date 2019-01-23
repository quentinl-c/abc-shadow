from .graph_model import GraphModel


class TwoInteractionsGraphModel(GraphModel):

    type_values = {0, 1, 2}

    def __init__(self, l1, l2, l12):
        self._l1 = l1
        self._l2 = l2
        self._l12 = l12

    @property
    def l1(self):
        return self._l1

    @property
    def l2(self):
        return self._l2

    @property
    def l12(self):
        return self._l12

    @l1.setter
    def l1(self, l1_new):
        self._l1 = l1_new

    @l2.setter
    def l2(self, l2_new):
        self._l2 = l2_new

    @l12.setter
    def l12(self, l12_new):
        self._l12 = l12_new

    def set_params(self, *args):

        if len(args) < 3:
            raise ValueError

        self.l1 = args[0]
        self.l2 = args[1]
        self.l12 = args[2]

    def evaluate_from_stats(self, *args):

        if len(args) < 3:
            raise ValueError

        u = self._l1 * args[0]
        u += self._l2 * args[1]
        u += self._l12 * args[2]

        return u

    def evaluate(self, sample):
        """Given a graph (sample),
        computes the energy function of the Model

        Arguments:
            sample {GraphWrapper} -- input sample

        Returns:
            float -- resulting energy
        """
        u = self._l1 * sample.get_edge_type_count(1)
        u += self._l2 * sample.get_edge_type_count(2)
        u += self._l12 * sample.get_diff_type_count()

        return u

    def get_local_energy(self, sample, edge, neigh):
        """Compute the local energy.

        Arguments:
            sample {GraphWrapper} -- Input sample
            edge {EdgeId} -- Edge identifier

        Returns:
            float -- Delta energy
        """

        edge_type = sample.get_edge_type(edge)

        if edge_type == 0:
            return 0

        if edge_type == 1:
            res = self._l1
        elif edge_type == 2:
            res = self._l2

        count = 0

        for t in neigh:
            if t != 0 and t != edge_type:
                count += 1

        res += self._l12 * count
        # if edge_type == 1:
        #     print(edge_type, res)
        return res

    @staticmethod
    def summary(results):
        data = dict()
        data['l1'] = [g.get_edge_type_count(1) for g in results]
        data['l2'] = [g.get_edge_type_count(2) for g in results]
        data['l12'] = [g.get_diff_type_count() for g in results]
        return data
