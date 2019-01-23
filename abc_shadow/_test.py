from abc_shadow.graph.graph_wrapper import GraphWrapper

G = GraphWrapper(10)

nodes = [(1, 8), (1,4) ,(4,6)]

for n in nodes:
    G.set_edge_type(n, 1)
