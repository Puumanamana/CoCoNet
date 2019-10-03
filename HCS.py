import networkx as nx
import numpy as np

"""Python implementation of basic HCS
Implementation of Highly Connected Subgraphs (HCS) clustering which is introduced by "Hartuv, E., & Shamir, R. (2000).
 A clustering algorithm based on graph connectivity. Information processing letters, 76(4-6), 175-18"
 
Based on NetworkX and Numpy
Notation:
    G = Graph
    E = Edge
    V = Vertex
    
    |V| = Number of Vertices in G
    |E| = Number of Edges in G
"""

def highly_connected(G, E, ratio=0.5):
    """Checks if the graph G is highly connected
    Highly connected means, that splitting the graph G into subgraphs needs more than 0.5*|V| edge deletions
    This definition can be found in Section 2 of the publication.
    :param G: Graph G
    :param E: Edges needed for splitting G
    :return: True if G is highly connected, otherwise False
    """

    return len(E) > len(G.nodes)*ratio


def remove_edges(G, E):
    """Removes all edges E from G
    Iterates over all edges in E and removes them from G
    :param G: Graph to remove edges from
    :param E: One or multiple Edges
    :return: Graph with edges removed
    """

    for edge in E:
        G.remove_edge(*edge)
    return G


def HCS(G, ratio=0.5):
    """Basic HCS Algorithm
    cluster labels, removed edges are stored in global variables
    :param G: Input graph
    :return: Either the input Graph if it is highly connected, otherwise a Graph composed of
    Subgraphs that build clusters
    """

    E = nx.algorithms.connectivity.cuts.minimum_edge_cut(G)

    if not highly_connected(G, E, ratio=ratio):
        G = remove_edges(G, E)
        sub_graphs = list(nx.connected_component_subgraphs(G))

        if len(sub_graphs) >= 2:
            H = [HCS(sub_graph, ratio=ratio) for sub_graph in sub_graphs]
            G = nx.algorithms.operators.all.compose_all(H)
    return G


def labelled_HCS(G, ratio=0.5):
    """
    Runs basic HCS and returns Cluster Labels
    :param G: Input graph
    :return: List of cluster assignments for the single vertices
    """

    _G = HCS(G, ratio=ratio)
    sub_graphs = nx.connected_component_subgraphs(_G)

    labels = np.zeros(shape=(len(G)), dtype=np.uint16)

    for _class, _cluster in enumerate(sub_graphs, 1):
        c = list(_cluster.nodes)
        labels[c] = _class

    return labels
