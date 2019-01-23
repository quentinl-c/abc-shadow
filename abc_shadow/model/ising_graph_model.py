from .binomial_edge_graph_model import BinomialEdgeGraphModel


class IsingGrapModel(BinomialEdgeGraphModel):

    def __init__(self, edge, homophily):
        super().__init__(edge)
        self._heffect = homophily

    @property
    def heffect(self):
        return self._heffect

    @heffect.setter
    def heffect(self, new_heffect):
        self._heffect = new_heffect

    def set_params(self, *args):

        if len(args) < 2:
            raise ValueError

        super().set_params(args[0])
        self._heffect = args[1]

    def evaluate(self, sample):
        edge_side = super().evaluate(sample)
        res = edge_side + self._heffect * sample.get_heffect()
        return res