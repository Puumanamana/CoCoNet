'''
Groups all the functions required to run the clustering steps
'''

import sys

import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm

import torch

import igraph
import leidenalg

from coconet.tools import run_if_not_exists

def compute_pairwise_comparisons(model, graph, handles,
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

    if bin_id < 0:
        contig_iter = tqdm(enumerate(contigs), total=len(contigs))
        contig_iter.bar_format = "{desc:<70} {percentage:3.0f}%|{bar:20}"
    else:
        contig_iter = enumerate(contigs[neighbors])

        if len(neighbors) > 100:
            contig_iter = tqdm(contig_iter, total=len(neighbors))
            contig_iter.bar_format = "{desc:<30} {percentage:3.0f}%|{bar:10}"
            contig_iter.set_description(f'Refining bin #{bin_id} ({len(neighbors)} contigs)')

    edges = {}

    for k, ctg in contig_iter:

        # graph.neighbors() returns the index of the contigs that are connected to ctg
        if bin_id == -1:
            # Case 1: we are computing the pre-graph
            # We limit the comparisons to the closest neighbors (precomputed)
            neighb_idx = neighbors[k]
        else:
            # Case 2: we are computing all remaining comparisons among contigs in a given bin
            neighb_idx = neighbors

        processed = np.isin(neighb_idx, graph.neighbors(ctg))

        neighb_k = contigs[neighb_idx][~processed][:int(max_neighbors)]

        if neighb_k.size == 0:
            continue

        if bin_id < 0:
            contig_iter.set_description(f'Contig #{k} - {len(neighb_k)} neighbors')

        x_ref = {k: torch.from_numpy(handle.get(ctg)[:][ref_idx]) for k, handle in handles.items()}

        # Compare the reference contig against all of its neighbors
        for other_ctg in neighb_k:
            if other_ctg == ctg or (other_ctg, ctg) in edges:
                continue

            x_other = {k: torch.from_numpy(handle.get(other_ctg)[:][other_idx])
                       for k, handle in handles.items()}

            probs = model.combine_repr(x_ref, x_other)['combined'].detach().numpy()[:, 0]
            edges[(ctg, other_ctg)] = sum(probs)

    if edges:
        prev_weights = graph.es['weight']
        graph.add_edges(list(edges.keys()))
        graph.es['weight'] = prev_weights + list(edges.values())

@run_if_not_exists()
def make_pregraph(model, features, output, force=False, **kw):
    '''
    Fill contig-contig adjacency matrix. For a given contig:
    - Extract neighbors
    - Make all n_frags**2 comparisons
    - Fill the value with the expected number of hits
    '''

    contigs = features['composition'].get_contigs()

    # Intersect neighbors for all features
    neighbors_each = {name: feature.get_neighbors() for (name, feature) in features.items()}
    neighbors = []

    for (n1, n2) in zip(*neighbors_each.values()):
        common_neighbors = n1[np.isin(n1, n2)]
        neighbors.append(common_neighbors)

    # Initialize graph
    graph = igraph.Graph()
    graph.add_vertices(contigs)
    graph.es['weight'] = []
    
    # Compute edges
    handles = {name: feature.get_handle('latent')
               for (name, feature) in features.items()}
    compute_pairwise_comparisons(model, graph, handles, neighbors=neighbors, **kw)

    for handle in handles.values():
        handle.close()
    
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
def iterate_clustering(model, repr_file, pre_graph_file,
                       singletons_file=None,
                       graph_file=None,
                       assignments_file=None,
                       n_frags=30,
                       theta=0.9,
                       gamma1=0.1, gamma2=0.75,
                       force=False,
                       logger=None):
    '''
    - Go through all clusters
    - Fill the pairwise comparisons within clusters
    - Re-cluster the clusters
    '''

    # Pre-clustering
    pre_graph = igraph.Graph.Read_Pickle(pre_graph_file)
    edge_threshold = theta * n_frags**2

    handles = {key: h5py.File(filename, 'r') for key, filename in repr_file.items()}
    contigs = np.array(list(handles['coverage'].keys()))
    get_communities(pre_graph, edge_threshold, gamma=gamma1)

    clusters = pre_graph.vs['cluster']

    mapping_id_ctg = pd.Series(pre_graph.vs.indices, index=pre_graph.vs['name'])

    # Refining the clusters
    contigs = []
    assignments = []
    clusters = np.unique(pre_graph.vs['cluster'])
    last_cluster = max(clusters)

    # Logging
    msg = f'Refining graph ({len(clusters)} clusters)'
    if logger is None: print(msg)
    else: logger.info(msg)
    
    for i, cluster in enumerate(clusters):
        
        if i > 0 and i // len(clusters) == 10:
            logger.debug(f'{i:,} clusters processed')

        contigs_c = pre_graph.vs.select(cluster=cluster)['name']
        contigs += contigs_c

        if len(contigs_c) < 3:
            assignments += [cluster] * len(contigs_c)
            continue

        # Compute all comparisons in this cluster
        compute_pairwise_comparisons(model, pre_graph, handles,
                                     neighbors=mapping_id_ctg[contigs_c].values,
                                     max_neighbors=5000,
                                     n_frags=n_frags,
                                     bin_id=cluster)
        # Find the leiden communities
        sub_graph = pre_graph.subgraph(contigs_c)
        get_communities(sub_graph, edge_threshold, gamma=gamma2)

        assignments += [x + last_cluster + 1 for x in sub_graph.vs['cluster']]
        last_cluster = max(assignments)

    # get_communities(pre_graph, edge_threshold, gamma=gamma2)
    # assignments = pd.Series(dict(zip(pre_graph.vs['name'], pre_graph.vs['cluster'])))
    # last_cluster = assignments.max()

    assignments = pd.Series(dict(zip(contigs, assignments)))

    # Add the rest of the contigs (singletons) set aside at the beginning
    ignored = pd.read_csv(singletons_file, sep='\t', usecols=['contigs'], index_col='contigs')
    ignored['clusters'] = np.arange(len(ignored)) + last_cluster + 1

    if np.intersect1d(ignored.index.values, pre_graph.vs['name']).size > 0:
        logger.error('Something went wrong: singleton contigs are already in the graph.')
        sys.exit()

    pre_graph.add_vertices(ignored.index.values)

    assignments = pd.concat([assignments, ignored.clusters]).loc[pre_graph.vs['name']]

    # Update the graph
    pre_graph.vs['cluster'] = assignments.tolist()
    pre_graph.write_pickle(graph_file)

    # Write the cluster in .csv format
    communities = pd.Series(pre_graph.vs['cluster'], index=pre_graph.vs['name'])
    communities.to_csv(assignments_file, header=False)
