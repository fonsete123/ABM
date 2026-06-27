# social network builders

import networkx as nx


# build_network
def build_net(kind, n, m, seed):
    if kind == "sw":
        return nx.watts_strogatz_graph(n, 2*m, 0.1, seed=seed)
    if kind == "er":
        return nx.erdos_renyi_graph(n, 2*m/n, seed=seed)
    return nx.barabasi_albert_graph(n, m, seed=seed)        # baseline, hubs
