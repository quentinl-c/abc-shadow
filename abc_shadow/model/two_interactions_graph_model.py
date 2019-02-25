from .graph_model import GraphModel


class TwoInteractionsGraphModel(GraphModel):

    type_values = {0, 1, 2}

    def __init__(self, *args):
        if len(args) != 4:
            raise ValueError

        super().__init__(*args)

    @property
    def l0(self):
        return self._params[0]

    @property
    def l1(self):
        return self._params[1]

    @property
    def l2(self):
        return self._params[2]

    @property
    def l12(self):
        return self._params[3]

    def set_params(self, *args):

        if len(args) != 4:
            raise ValueError
        super().set_params(*args)

    def evaluate_from_stats(self, *args):

        if len(args) != 4:
            raise ValueError

        return super().evaluate_from_stats(*args)

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
        u += self.l12 * sample.get_repulsion_count(excluded_lables=[0])

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
        data['l1'] = [g.get_edge_type_count(1) for g in results]
        data['l2'] = [g.get_edge_type_count(2) for g in results]
        data['l12'] = [g.get_repulsion_count(exluded_labels=[0]) for g in results]

        return data
