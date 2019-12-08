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


def compute_pairwise_comparisons(model, contigs, handles,
                                 init_matrix=None, neighbors=None,
                                 n_frags=30, max_neighbors=100):
    '''
    Compare each contig in [contigs] with its [neighbors]
    by running the neural network for each fragment combinations
    between the contig pair
    '''

    ref_idx, other_idx = (np.repeat(np.arange(n_frags), n_frags),
                          np.tile(np.arange(n_frags), n_frags))

    adjacency_matrix = init_matrix

    # Start from scratch or from a provided expected hits matrix
    if init_matrix is None:
        adjacency_matrix = np.identity(len(contigs))*(n_frags**2+1) \
            - np.ones((len(contigs), len(contigs)))

    for k, ctg in enumerate(contigs):

        print('Processed contigs: {:,}/{:,}'.format(k, len(contigs)), end='\r')

        x_ref = {key: torch.from_numpy(np.array(handle.get(ctg)[:])[ref_idx])
                 for key, handle in handles.items()}

        if neighbors is None:
            # We use all contigs that where not already processed
            scores = adjacency_matrix[k, :]
            new_neighbors_k = np.arange(len(contigs))[scores < 0]
        else:
            scores = adjacency_matrix[k, neighbors[k]]
            new_neighbors_k = neighbors[k][scores < 0][:max_neighbors]

        for n_i in new_neighbors_k:
            x_other = {key: torch.from_numpy(
                np.array(
                    handle.get(contigs[n_i])[:]
                )[other_idx])
                       for key, handle in handles.items()}
            probs = model.combine_repr(x_ref, x_other)['combined'].detach().numpy()
            # Get number of expected matches
            adjacency_matrix[k, n_i] = sum(probs)
            adjacency_matrix[n_i, k] = adjacency_matrix[k, n_i]

    return adjacency_matrix

@run_if_not_exists()
def fill_adjacency_matrix(model, latent_repr, output, **kw):
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

    adjacency_matrix = compute_pairwise_comparisons(
        model, contigs, handles, neighbors=neighbors, **kw)

    np.save(output, adjacency_matrix)

def get_communities(adjacency_matrix, contigs, gamma=0.5, truth_sep='|'):
    '''
    Cluster using with either Louvain or Leiden algorithm defined in config
    Use the adjacency matrix previously filled
    '''

    graph = igraph.Graph.Adjacency(adjacency_matrix.tolist())
    optimiser = leidenalg.Optimiser()

    partition = leidenalg.CPMVertexPartition(graph, resolution_parameter=gamma)
    optimiser.optimise_partition(partition)

    communities = (pd.Series(dict(enumerate(partition)))
                   .explode()
                   .sort_values()
                   .index)

    assignments = pd.DataFrame({
        'clusters': communities,
        'truth': [x.split(truth_sep)[0] for x in contigs],
        'contigs': contigs
    }).set_index('contigs')

    assignments.clusters = pd.factorize(assignments.clusters)[0]
    assignments.truth = pd.factorize(assignments.truth)[0]

    return assignments

@run_if_not_exists(keys=('refined_assignments', 'refined_adj_mat'))
def iterate_clustering(model, repr_file, adj_mat_file,
                       singletons_file=None,
                       assignments_file='assignments.csv',
                       refined_adj_mat_file='refined_adjacency_matrix.npy',
                       refined_assignments_file='refined_assignments.csv',
                       n_frags=30,
                       hits_threshold=0.9,
                       gamma1=0.1,
                       gamma2=0.75,
                       max_neighbors=100):
    '''
    - Go through all clusters
    - Fill the pairwise comparisons within clusters
    - Re-cluster the clusters
    '''

    # Pre-clustering
    adjacency_matrix = np.load(adj_mat_file)
    edge_threshold = hits_threshold * n_frags**2

    handles = {key: h5py.File(filename, 'r') for key, filename in repr_file.items()}
    contigs = np.array(list(handles['coverage'].keys()))

    communities = get_communities(
        (adjacency_matrix > edge_threshold).astype(int),
        contigs,
        gamma=gamma1
    )

    ignored = pd.read_csv(singletons_file, sep='\t', usecols=['contigs'], index_col='contigs')
    ignored['clusters'] = np.arange(len(ignored)) + communities.clusters.max() + 1
    ignored['truth'] = -1

    communities = pd.concat([communities, ignored])
    communities.truth = pd.factorize(communities.index.str.split('|').str[0])[0]

    communities.to_csv(assignments_file)

    # Refining the clusters
    communities['ctg_index'] = np.arange(communities.shape[0])
    # Get the contigs indices from each cluster
    clusters = communities.groupby('clusters')['ctg_index'].agg(list)

    for ctg_indices in clusters.values:
        if len(ctg_indices) < 3:
            continue

        # Compute remaining pairwise comparisons
        adjacency_submatrix = compute_pairwise_comparisons(
            model, contigs[ctg_indices], handles,
            init_matrix=adjacency_matrix[ctg_indices, :][:, ctg_indices],
            n_frags=n_frags, max_neighbors=max_neighbors
        )
        mask = np.isin(np.arange(adjacency_matrix.shape[0]), ctg_indices)
        mask = mask.reshape(-1, 1).dot(mask.reshape(1, -1))

        adjacency_matrix[mask] = adjacency_submatrix.flatten()

        if not np.all(adjacency_submatrix > edge_threshold):
            sub_communities = get_communities(
                (adjacency_submatrix > edge_threshold).astype(int),
                contigs[ctg_indices],
                gamma=gamma2,
            )
            communities.loc[sub_communities.index, 'clusters'] = (1 + sub_communities.clusters
                                                                  + communities.clusters.max())
    np.save(refined_adj_mat_file, adjacency_matrix)

    communities.clusters = pd.factorize(communities.clusters)[0]
    communities.drop('ctg_index', axis=1, inplace=True)

    communities.to_csv(refined_assignments_file)
