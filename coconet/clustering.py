'''
Groups all the functions required to run the clustering steps
'''

import numpy as np
import pandas as pd
import logging

import torch

import igraph
import leidenalg

from coconet.tools import run_if_not_exists


logger = logging.getLogger('clustering')

def compute_pairwise_comparisons(model, graph, handles,
                                 vote_threshold=None,
                                 neighbors=None, max_neighbors=100,
                                 n_frags=30, bin_id=-1):
    '''
    Compare each contig in [contigs] with its [neighbors]
    by running the neural network for each fragment combinations
    between the contig pair
    '''

    ref_idx, other_idx = (np.repeat(np.arange(n_frags), n_frags),
                          np.tile(np.arange(n_frags), n_frags))

    contigs = np.array(graph.vs['name'])

    edges = {}

    for k, ctg in enumerate(contigs):
        # Case 1: we are computing the pre-graph --> We limit the comparisons to the closest neighbors
        if bin_id == -1:
            neighb_idx = neighbors[k]
        # Case 2: we are computing all remaining comparisons among contigs in a given bin
        else:
            neighb_idx = neighbors

        processed = np.isin(neighb_idx, graph.neighbors(ctg))

        neighb_k = contigs[neighb_idx][~processed][:int(max_neighbors)]

        if neighb_k.size == 0:
            continue

        x_ref = {name: torch.from_numpy(handle[ctg][:][ref_idx]) for name, handle in handles}

        if len(handles) == 1: # Only one feature type
            feature = list(x_ref)[0]
            x_ref = x_ref[feature]

        # Compare the reference contig against all of its neighbors
        for other_ctg in neighb_k:
            if other_ctg == ctg or (other_ctg, ctg) in edges:
                continue

            x_other = {name: torch.from_numpy(handle[other_ctg][:][other_idx])
                       for name, handle in handles}

            if len(handles) == 1: # Only one feature type
                x_other = x_other[feature]
                probs = model.combine_repr(x_ref, x_other).detach().numpy()[:, 0]
            else:
                probs = model.combine_repr(x_ref, x_other)['combined'].detach().numpy()[:, 0]

            if vote_threshold is not None:
                probs = probs > vote_threshold

            edges[(ctg, other_ctg)] = sum(probs)

    if edges:
        prev_weights = graph.es['weight']
        graph.add_edges(list(edges.keys()))
        graph.es['weight'] = prev_weights + list(edges.values())

@run_if_not_exists()
def make_pregraph(model, features, output, **kw):
    '''
    Fill contig-contig adjacency matrix. For a given contig:
    - Extract neighbors
    - Make all n_frags**2 comparisons
    - Fill the value with the expected number of hits
    '''

    contigs = features[0].get_contigs()

    # Intersect neighbors for all features
    neighbors_each = [feature.get_neighbors() for feature in features]

    if len(features) > 1:
        neighbors = []
        for (n1, n2) in zip(*neighbors_each):
            common_neighbors = n1[np.isin(n1, n2)]
            neighbors.append(common_neighbors)
    else:
        neighbors = neighbors_each[0]

    # Initialize graph
    graph = igraph.Graph()
    graph.add_vertices(contigs)
    graph.es['weight'] = []

    # Compute edges
    handles = [(feature.name, feature.get_handle('latent')) for feature in features]
    compute_pairwise_comparisons(model, graph, handles, neighbors=neighbors, **kw)

    for handle in handles:
        handle[1].close()

    # Save pre-graph
    graph.write_pickle(output)

def get_communities(graph, threshold, gamma=0.5):
    '''
    Cluster using with the Leiden algorithm with parameter gamma
    Use the graph previously filled
    '''

    cluster_graph = graph.copy()
    cluster_graph.es.select(weight_lt=threshold).delete()

    optimiser = leidenalg.Optimiser()

    partition = leidenalg.CPMVertexPartition(cluster_graph, resolution_parameter=gamma)
    optimiser.optimise_partition(partition)

    communities = pd.Series(dict(enumerate(partition))).explode()

    # Set the "cluster" attribute
    graph.vs['cluster'] = communities.sort_values().index

@run_if_not_exists(keys=('assignments_file', 'graph_file'))
def iterate_clustering(model,
                       features,
                       pre_graph_file,
                       singletons_file=None,
                       graph_file=None,
                       assignments_file=None,
                       n_frags=30,
                       theta=0.9,
                       vote_threshold=None,
                       gamma1=0.1, gamma2=0.75):
    '''
    - Go through all clusters
    - Fill the pairwise comparisons within clusters
    - Re-cluster the clusters
    '''

    # Pre-clustering
    pre_graph = igraph.Graph.Read_Pickle(pre_graph_file)
    edge_threshold = theta * n_frags**2

    get_communities(pre_graph, edge_threshold, gamma=gamma1)

    mapping_id_ctg = pd.Series(pre_graph.vs.indices, index=pre_graph.vs['name'])

    # Refining the clusters
    contigs = []
    assignments = []
    clusters = np.unique(pre_graph.vs['cluster'])
    last_cluster = max(clusters)

    logger.info(f'Processing {clusters.size} clusters')

    handles = [(feature.name, feature.get_handle('latent')) for feature in features]

    for i, cluster in enumerate(clusters):

        contigs_c = pre_graph.vs.select(cluster=cluster)['name']
        contigs += contigs_c

        if len(contigs_c) <= 2:
            assignments += [cluster] * len(contigs_c)
            continue

        if len(contigs_c) > 50:
            logger.info(f"Processing cluster #{i} ({len(contigs_c)} contigs)")

        elif i > 0 and i % (len(clusters)//5) == 0:
            logger.info(f'{i:,} clusters processed')

        # Compute all comparisons in this cluster
        compute_pairwise_comparisons(model, pre_graph, handles,
                                     neighbors=mapping_id_ctg[contigs_c].values,
                                     max_neighbors=5000,
                                     n_frags=n_frags,
                                     vote_threshold=vote_threshold,
                                     bin_id=cluster)
        # Find the leiden communities
        sub_graph = pre_graph.subgraph(contigs_c)
        get_communities(sub_graph, edge_threshold, gamma=gamma2)

        assignments += [x + last_cluster + 1 for x in sub_graph.vs['cluster']]
        last_cluster = max(assignments)

    for _, handle in handles:
        handle.close()

    assignments = pd.Series(dict(zip(contigs, assignments)))

    # Add the rest of the contigs (singletons) set aside at the beginning
    ignored = pd.read_csv(singletons_file, sep='\t', usecols=['contigs'], index_col='contigs')
    ignored['clusters'] = np.arange(len(ignored)) + last_cluster + 1

    if np.intersect1d(ignored.index.values, pre_graph.vs['name']).size > 0:
        logger.error('Something went wrong: singleton contigs are already in the graph.')
        raise RuntimeError

    pre_graph.add_vertices(ignored.index.values)

    assignments = pd.concat([assignments, ignored.clusters]).loc[pre_graph.vs['name']]

    # Update the graph
    pre_graph.vs['cluster'] = assignments.tolist()
    pre_graph.write_pickle(graph_file)

    # Write the cluster in .csv format
    communities = pd.Series(pre_graph.vs['cluster'], index=pre_graph.vs['name'])
    communities.to_csv(assignments_file, header=False)
