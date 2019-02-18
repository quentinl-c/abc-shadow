from .graph_model import GraphModel


class TwoInteractionsGraphModel(GraphModel):

    type_values = {0, 1, 2}

    def __init__(self, l0, l1, l2, l12):
        self._l0 = l0
        self._l1 = l1
        self._l2 = l2
        self._l12 = l12

    @property
    def l0(self):
        return self._l0

    @property
    def l1(self):
        return self._l1

    @property
    def l2(self):
        return self._l2

    @property
    def l12(self):
        return self._l12

    @l0.setter
    def l0(self, l0_new):
        self._l0 = l0_new

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

        if len(args) < 4:
            raise ValueError
        self.l0 = args[0]
        self.l1 = args[1]
        self.l2 = args[2]
        self.l12 = args[3]

    def evaluate_from_stats(self, *args):

        if len(args) < 4:
            raise ValueError

        u = self.l0 * args[0]
        u += self.l1 * args[1]
        u += self.l2 * args[2]
        u += self.l12 * args[3]

        return u

    def evaluate(self, sample):
        """Given a graph (sample),
        computes the energy function of the Model

        Arguments:
            sample {GraphWrapper} -- input sample

        Returns:
            float -- resulting energy
        """
        u = self.l0 * sample.get_edge_count()
        u += self.l1 * sample.get_edge_type_count(1)
        u += self.l2 * sample.get_edge_type_count(2)
        u += self.l12 * sample.get_diff_type_count()

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

        res = self.l0

        if edge_type == 1:
            res += self.l1
        elif edge_type == 2:
            res += self.l2

        count = 0
        for t in neigh:
            n_label = sample.vertex[t]
            if n_label != 0 and n_label != edge_type:
                count += 1

        res += self.l12 * count
        # if edge_type == 1:
        #     print(edge_type, res)
        return res

    @staticmethod
    def summary(results):
        data = dict()
        data['l0'] = [g.get_edge_count() for g in results]
        # data['-l0'] = [g.get_none_edge_count() for g in results]
        data['l1'] = [g.get_edge_type_count(1) for g in results]
        data['l2'] = [g.get_edge_type_count(2) for g in results]
        data['l12'] = [g.get_diff_type_count() for g in results]

        return data
