'''
Clustering utils:
- save_repr_all()
- fill_adjacency_matrix()
- get_communities()
- iterate_clustering()
'''

import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm

import torch
from sklearn.metrics.pairwise import euclidean_distances

import igraph
import leidenalg

from coconet.tools import run_if_not_exists

def get_neighbors(file_h5):
    '''
    - Extract the latent representation of each contig's fragments in the h5 file
    - Extract the center, radius and pairwise distances between centers
    - For each contig, return the neighbors
    '''

    handle = h5py.File(file_h5, 'r')

    # data.shape = ( n_contigs, n_frags, latent_dim )
    data = np.stack([np.array(handle.get(ctg)[:]) for ctg in handle.keys()])
    # center of each contigs (n_contigs, latent_dim)
    contig_centers = np.mean(data, axis=1)
    # pairwise distances between contig centers
    pairwise_distances = euclidean_distances(contig_centers)
    # distance of each fragment to its center
    distance_to_resp_center = np.sqrt(np.sum(
        (data - contig_centers[:, None, :])**2, axis=2
    ))
    # radius of each contig (mean+2*std)(distance) from fragment to center
    radii = np.mean(distance_to_resp_center, axis=1) + 2*np.std(distance_to_resp_center, axis=1)
    # Condition: neighbors need to be within [radius] units from the center
    within_range = pairwise_distances < radii.reshape(-1, 1)

    # get neighbors indices and sort them by distance
    indices = np.arange(len(radii))
    neighbors_ordered = [indices[wr][pairwise_distances[i, wr].argsort()]
                         for i, wr in enumerate(within_range)]

    return neighbors_ordered

def compute_pairwise_comparisons(model, graph, handles, contigs=None,
                                 neighbors=None, max_neighbors=100,
                                 n_frags=30, bin_id=-1):
    '''
    Compare each contig in [contigs] with its [neighbors]
    by running the neural network for each fragment combinations
    between the contig pair
    '''

    ref_idx, other_idx = (np.repeat(np.arange(n_frags), n_frags),
                          np.tile(np.arange(n_frags), n_frags))

    if contigs is None:
        contigs = [v['name'] for v in graph.vs]

    contig_iter = tqdm(enumerate(contigs), ncols=100, total=len(contigs))

    if bin_id < 0:
        contig_iter.bar_format = "{desc:<70} {percentage:3.0f}%|{bar}"
    else:
        contig_iter.bar_format = "{desc:<40} {percentage:3.0f}%|{bar}"
        contig_iter.set_description(
            "Refining bin #{} ({} contigs)".format(bin_id, len(contigs))
        )

    edges = []
    weights = []

    for k, ctg in contig_iter:
        # graph.neighbors() returns the index of the contigs that are connected to ctg
        if bin_id == -1:
            processed = np.isin(neighbors[k], graph.neighbors(ctg))
            new_neighbors_k = neighbors[k][~processed][:max_neighbors]
        else:
            processed = np.isin(neighbors, graph.neighbors(ctg))
            new_neighbors_k = np.arange(len(neighbors))[~processed][:max_neighbors]

        if new_neighbors_k.size == 0:
            continue

        if bin_id < 0:
            contig_iter.set_description(
                "Contig #{} - Computing comparison with neighbors ({} contigs)"
                .format(k, len(new_neighbors_k))
            )

        x_ref = {}
        for key, handle in handles.items():
            x_npy = np.tile(
                handle.get(ctg)[:][ref_idx],
                (len(new_neighbors_k), 1)
            )
            x_ref[key] = torch.from_numpy(x_npy)

        x_other = {}
        for key, handle in handles.items():
            x_npy = np.vstack(
                [handle.get(contigs[n_i])[:][other_idx]
                 for n_i in new_neighbors_k]
            )
            x_other[key] = torch.from_numpy(x_npy)

        probs = model.combine_repr(x_ref, x_other)['combined'].detach().numpy()[:, 0]
        probs = np.convolve(probs, np.ones(n_frags**2), 'valid')[::n_frags**2]

        for n_i, p_i in zip(new_neighbors_k, probs):
            edges.append((ctg, contigs[n_i]))
            weights.append(p_i)

    if edges:
        prev_weights = graph.es['weight']
        graph.add_edges(edges)
        graph.es['weight'] = prev_weights + weights

@run_if_not_exists()
def make_pregraph(model, latent_repr, output, **kw):
    '''
    Fill contig-contig adjacency matrix. For a given contig:
    - Extract neighbors
    - Make all n_frags**2 comparisons
    - Fill the value with the expected number of hits
    '''

    if output.is_file():
        return

    handles = {key: h5py.File(filename, 'r') for key, filename in latent_repr.items()}
    contigs = np.array(list(handles['coverage'].keys()))

    # Get neighbors for coverage feature
    neighbors = get_neighbors(latent_repr['coverage'])

    # Intersect with neighbors for composition feature
    for i, n_i in enumerate(get_neighbors(latent_repr['composition'])):
        neighbors[i] = np.intersect1d(neighbors[i], n_i)

    # Initialize graph
    graph = igraph.Graph()
    graph.add_vertices(contigs)
    graph.es['weight'] = []
    # Compute edges
    compute_pairwise_comparisons(model, graph, handles, neighbors=neighbors, **kw)
    # Save pre-graph
    graph.write_pickle(output)

def get_communities(graph, threshold, gamma=0.5):
    '''
    Cluster using with either Louvain or Leiden algorithm defined in config
    Use the adjacency matrix previously filled
    '''

    cluster_graph = graph.copy()
    cluster_graph.es.select(weight_lt=threshold).delete()

    optimiser = leidenalg.Optimiser()

    partition = leidenalg.CPMVertexPartition(cluster_graph, resolution_parameter=gamma)
    optimiser.optimise_partition(partition)

    communities = pd.Series(dict(enumerate(partition))).explode().sort_values()

    graph.vs['cluster'] = communities.index

@run_if_not_exists(keys=('assignments_file', 'graph_file'))
def iterate_clustering(model, repr_file, pre_graph_file,
                       singletons_file=None,
                       graph_file=None,
                       assignments_file=None,
                       n_frags=30,
                       hits_threshold=0.9,
                       gamma1=0.1, gamma2=0.75):
    '''
    - Go through all clusters
    - Fill the pairwise comparisons within clusters
    - Re-cluster the clusters
    '''

    # Pre-clustering
    pre_graph = igraph.Graph.Read_Pickle(pre_graph_file)
    edge_threshold = hits_threshold * n_frags**2

    handles = {key: h5py.File(filename, 'r') for key, filename in repr_file.items()}
    contigs = np.array(list(handles['coverage'].keys()))
    get_communities(pre_graph, edge_threshold, gamma=gamma1)

    clusters = pre_graph.vs['cluster']
    ignored = pd.read_csv(singletons_file, sep='\t', usecols=['contigs'], index_col='contigs')
    ignored['clusters'] = np.arange(len(ignored)) + max(clusters) + 1

    if np.intersect1d(ignored.index.values, pre_graph.vs['name']).size == 0:
        pre_graph.add_vertices(ignored.index.values)
        pre_graph.vs['cluster'] = clusters + ignored.clusters.tolist()

    mapping_id_ctg = pd.Series(pre_graph.vs.indices, index=pre_graph.vs['name'])

    # Refining the clusters
    contigs = []
    assignments = []
    clusters = np.unique(pre_graph.vs['cluster'])
    last_cluster = max(clusters)

    print('Refining graph ({} clusters)'.format(len(clusters)))
    for cluster in clusters:
        contigs_c = pre_graph.vs.select(cluster=cluster)['name']
        contigs += contigs_c

        if len(contigs_c) < 3:
            assignments += [cluster] * len(contigs_c)
            continue

        # Compute all comparisons in this cluster
        compute_pairwise_comparisons(model, pre_graph, handles, contigs=np.array(contigs_c),
                                     neighbors=mapping_id_ctg[contigs_c].values, max_neighbors=5000,
                                     n_frags=n_frags, bin_id=cluster)
        # Find the leiden communities
        sub_graph = pre_graph.subgraph(contigs_c)
        get_communities(sub_graph, edge_threshold, gamma=gamma2)

        assignments += [x + last_cluster + 1 for x in sub_graph.vs['cluster']]
        last_cluster = max(assignments)

    assignments = pd.Series(dict(zip(contigs, assignments)))
    pre_graph.vs['cluster'] = assignments.loc[pre_graph.vs['name']].values
    pre_graph.write_pickle(graph_file)
    get_communities(pre_graph, edge_threshold, gamma=gamma2)

    communities = pd.Series(pre_graph.vs['cluster'], index=pre_graph.vs['name'])
    communities.to_csv(assignments_file, header=False)
