import networkx as nx
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator
import matplotlib.pyplot as plt


def build_graph(W, gas_columns, fault_types, all_columns, threshold=0.1):
    weighted_edges = []
    for src in fault_types:
        for tgt in gas_columns:
            weight = W[all_columns.index(src), all_columns.index(tgt)]
            if abs(weight) > threshold:
                weighted_edges.append((src, tgt, weight))

    G = nx.DiGraph()
    G.add_weighted_edges_from(weighted_edges)

    while not nx.is_directed_acyclic_graph(G):
        cycle = list(nx.find_cycle(G))
        min_edge = min(cycle, key=lambda edge: abs(G.edges[edge]['weight']))
        G.remove_edge(*min_edge[:2])

    return G


def visualize_graph(G, path='causal_dag.png'):
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_color='lightblue',
            node_size=2000, font_size=10, edge_color='gray')
    edge_labels = {(u, v): f"{d['weight']:.3f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, label_pos=0.3)
    plt.title("Learned Causal DAG with Edge Weights")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.show()


def fit_causal_network(G, df, all_columns):
    model = BayesianNetwork(G.edges())
    model.fit(df[all_columns], estimator=BayesianEstimator, prior_type='BDeu')
    return model
