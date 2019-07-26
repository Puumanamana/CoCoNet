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

def highly_connected(G, E):
    """Checks if the graph G is highly connected
    Highly connected means, that splitting the graph G into subgraphs needs more than 0.5*|V| edge deletions
    This definition can be found in Section 2 of the publication.
    :param G: Graph G
    :param E: Edges needed for splitting G
    :return: True if G is highly connected, otherwise False
    """

    return len(E) > len(G.nodes) / 2


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


def HCS(G):
    """Basic HCS Algorithm
    cluster labels, removed edges are stored in global variables
    :param G: Input graph
    :return: Either the input Graph if it is highly connected, otherwise a Graph composed of
    Subgraphs that build clusters
    """

    E = nx.algorithms.connectivity.cuts.minimum_edge_cut(G)

    if not highly_connected(G, E):
        G = remove_edges(G, E)
        sub_graphs = list(nx.connected_component_subgraphs(G))

        if len(sub_graphs) == 2:
            H = HCS(sub_graphs[0])
            _H = HCS(sub_graphs[1])

            G = nx.compose(H, _H)
    return G

def labelled_HCS(G,start=1):
    """
    Runs basic HCS and returns Cluster Labels
    :param G: Input graph
    :return: List of cluster assignments for the single vertices
    """

    _G = HCS(G)
    sub_graphs = nx.connected_component_subgraphs(_G)

    nodes = np.zeros(shape=(len(G)), dtype="<U128")
    labels = np.zeros(shape=(len(G)), dtype=np.uint16)
    current_pos = 0

    for _class, _cluster in enumerate(sub_graphs, start):
        c = list(_cluster.nodes)
        nodes[current_pos:current_pos+len(c)] = c
        labels[current_pos:current_pos+len(c)] = _class
        current_pos += len(c)

    return nodes,labels

def not_connected_HCS(G):

    k = 0
    nodes = np.zeros(shape=(len(G)), dtype="<U128")
    labels = np.zeros(shape=(len(G)), dtype=np.uint16)

    n_subgraphs = sum(1 for _ in nx.connected_component_subgraphs(G))
    n_processed = 0
    
    
    for i,G_i in enumerate(nx.connected_component_subgraphs(G)):
        print("Processed {} nodes - Subgraph #{}/{}: {} nodes".format(n_processed,i,n_subgraphs,len(G_i)))
        
        if len(G_i) > 1:
            nodes_i, labels_i = labelled_HCS(G_i,start=k)
        else:
            nodes_i, labels_i = (list(G_i.nodes)[0], k)

        nodes[n_processed:n_processed+len(G_i)] = nodes_i
        labels[n_processed:n_processed+len(G_i)] = labels_i
        
        n_processed += len(G_i)
        k = labels.max()+1

    return nodes,labels

if __name__ == '__main__':

    adjacency_matrix = np.load('output_data/sim_5_only_cov/adjacency_matrix_nf30.npy')
    adjacency_matrix[adjacency_matrix < 0] = 0

    threshold = 0.95*900

    G = nx.from_numpy_matrix((adjacency_matrix>threshold).astype(int))

    res = not_connected_HCS(G)

    import ipdb;ipdb.set_trace()
