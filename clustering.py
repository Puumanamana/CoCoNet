'''
Clustering utils:
- save_repr_all()
- fill_adjacency_matrix()
- get_communities()
- iterate_clustering()
'''

import os
import numpy as np
import pandas as pd
import h5py

import torch
from sklearn.metrics.pairwise import euclidean_distances

import networkx as nx
import community
import igraph
import leidenalg

from Bio import SeqIO
from progressbar import progressbar

from tools import get_kmer_frequency, avg_window

def save_repr_all(model, config):
    '''
    - Calculate intermediate representation for all fragments of all contigs
    - Save it in a .h5 file
    '''

    cov_h5 = h5py.File(config.inputs['filtered']['coverage_h5'], 'r')

    repr_h5 = {key: h5py.File(filename, 'w') for key, filename in config.outputs['repr'].items()}

    n_frags = config.clustering['n_frags']

    for contig in progressbar(SeqIO.parse(config.inputs['filtered']['fasta'], "fasta"),
                              max_value=len(cov_h5)):
        step = int((len(contig)-config.fl) / n_frags)

        fragment_boundaries = [(step*i, step*i+config.fl) for i in range(n_frags)]

        x_composition = torch.from_numpy(np.stack(
            [get_kmer_frequency(str(contig.seq)[start:stop],
                                kmer=config.kmer, rc=config.rc)
             for (start, stop) in fragment_boundaries]
        ).astype(np.float32)) # Shape = (n_frags, 4**k)

        fragment_slices = np.array([np.arange(start, stop)
                                    for (start, stop) in fragment_boundaries])
        coverage_genome = np.array(cov_h5.get(contig.id)[:]).astype(np.float32)[:, fragment_slices]
        coverage_genome = np.swapaxes(coverage_genome, 1, 0)

        x_coverage = torch.from_numpy(
            np.apply_along_axis(
                lambda x: avg_window(x, config.wsize, config.wstep), 2, coverage_genome
            ).astype(np.float32))

        x_repr = model.compute_repr(x_composition, x_coverage)

        for key, handle in repr_h5.items():
            handle.create_dataset(contig.id, data=x_repr[key].detach().numpy(), dtype=np.float32)

    for handle in repr_h5.values():
        handle.close()

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

def compute_pairwise_comparisons(config, model, contigs, handles,
                                 init_matrix=None, neighbors=None):
    '''
    Compare each contig in [contigs] with its [neighbors]
    by running the neural network for each fragment combinations
    between the contig pair
    '''
    n_frags = config.clustering['n_frags']

    ref_idx, other_idx = (np.repeat(np.arange(n_frags), n_frags),
                          np.tile(np.arange(n_frags), n_frags))

    adjacency_matrix = init_matrix

    # Start from scratch or from a provided expected hits matrix
    if init_matrix is None:
        adjacency_matrix = np.identity(len(contigs))*(n_frags**2+1) \
            - np.ones((len(contigs), len(contigs)))

    for k, ctg in progressbar(enumerate(contigs), max_value=len(contigs)):

        x_ref = {key: torch.from_numpy(np.array(handle.get(ctg)[:])[ref_idx])
                 for key, handle in handles.items()}

        if neighbors is None:
            # We use all contigs that where not already processed
            scores = adjacency_matrix[k, :]
            new_neighbors_k = np.arange(len(contigs))[scores < 0]
        else:
            scores = adjacency_matrix[k, neighbors[k]]
            new_neighbors_k = neighbors[k][scores < 0][:config.clustering['max_neighbors']]

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

def fill_adjacency_matrix(model, config):
    '''
    Fill contig-contig adjacency matrix. For a given contig:
    - Extract neighbors
    - Make all n_frags**2 comparisons
    - Fill the value with the expected number of hits
    '''

    if os.path.exists(config.outputs['clustering']['adjacency_matrix']):
        return

    handles = {key: h5py.File(filename) for key, filename in config.outputs['repr'].items()}
    contigs = np.array(list(handles['coverage'].keys()))

    # Get neighbors for coverage feature
    neighbors = get_neighbors(config.outputs['repr']['coverage'])

    # Intersect with neighbors for composition feature
    for i, n_i in enumerate(get_neighbors(config.outputs['repr']['composition'])):
        neighbors[i] = np.intersect1d(neighbors[i], n_i)

    adjacency_matrix = compute_pairwise_comparisons(
        config, model, contigs, handles, neighbors=neighbors)

    np.save(config.outputs['clustering']['adjacency_matrix'], adjacency_matrix)

def check_partitions(communities, adjacency_matrix, gamma, debug=False):
    def count_mean_edges(s):
        sub_adj = adjacency_matrix[s, :][:, s]
        return np.sum(sub_adj) / sub_adj.size

    clusters = (communities.reset_index()
                .groupby('clusters')['index']
                .agg(conn=count_mean_edges, cize=len, indices=list)
                .sort_values(by='conn'))

    if clusters.conn.min() < gamma:
        print('Suspicious cluster: {}'.format(clusters.conn.idxmin()))

        if debug:
            import ipdb;ipdb.set_trace()

def get_communities(adjacency_matrix, contigs, gamma=0.5, truth_sep='|', debug=False):
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

    check_partitions(assignments.reset_index(), adjacency_matrix, gamma, debug=debug)

    assignments.clusters = pd.factorize(assignments.clusters)[0]
    assignments.truth = pd.factorize(assignments.truth)[0]

    return assignments

def iterate_clustering(model, config):
    '''
    - Go through all clusters
    - Fill the pairwise comparisons within clusters
    - Re-cluster the clusters
    '''

    # Pre-clustering
    adjacency_matrix = np.load(config.outputs['clustering']['adjacency_matrix'])
    edge_threshold = config.clustering['hits_threshold'] * config.clustering['n_frags']**2

    handles = {key: h5py.File(filename) for key, filename in config.outputs['repr'].items()}
    contigs = np.array(list(handles['coverage'].keys()))

    communities = get_communities(
        (adjacency_matrix > edge_threshold).astype(int),
        contigs,
        gamma=config.clustering['gamma_1'],
        debug=True
    )

    communities.to_csv(config.outputs['clustering']['assignments'])

    # Refining the clusters
    communities['ctg_index'] = np.arange(communities.shape[0])
    # Get the contigs indices from each cluster
    clusters = communities.groupby('clusters')['ctg_index'].agg(list)

    for ctg_indices in clusters.values:
        if len(ctg_indices) < 3:
            continue

        # Compute remaining pairwise comparisons
        adjacency_submatrix = compute_pairwise_comparisons(
            config, model, contigs[ctg_indices], handles,
            init_matrix=adjacency_matrix[ctg_indices, :][:, ctg_indices]
        )
        mask = np.isin(np.arange(adjacency_matrix.shape[0]), ctg_indices)
        mask = mask.reshape(-1, 1).dot(mask.reshape(1, -1))

        adjacency_matrix[mask] = adjacency_submatrix.flatten()

        if not np.all(adjacency_submatrix > edge_threshold):
            sub_communities = get_communities(
                (adjacency_submatrix > edge_threshold).astype(int),
                contigs[ctg_indices],
                gamma=config.clustering['gamma_2'],
            )
            communities.loc[sub_communities.index, 'clusters'] = (1 + sub_communities.clusters
                                                                  + communities.clusters.max())
    np.save(config.outputs['clustering']['refined_adjacency_matrix'], adjacency_matrix)

    communities.clusters = pd.factorize(communities.clusters)[0]
    communities.drop('ctg_index', axis=1, inplace=True)

    communities.to_csv(config.outputs['clustering']['refined_assignments'])


def plot_communities(adj_sub, contig_names, sub_communities, louvain, thresh):
    '''
    Temporary metrics for analysis
    '''
    from sklearn.metrics import adjusted_rand_score
    from HCS import labelled_HCS

    adj_bin = (adj_sub > thresh).astype(float)

    res = np.array(labelled_HCS(nx.from_numpy_matrix(adj_bin), ratio=0.5))
    truth = pd.factorize([x.split('|')[0] for x in contig_names])[0]

    info = pd.DataFrame({
        'leiden': sub_communities.clusters.values,
        'louvain': louvain.values,
        'HCS': res,
        'truth': truth
    })
    print(info)

    print(adjusted_rand_score(truth, sub_communities.clusters),
          adjusted_rand_score(truth, louvain),
          adjusted_rand_score(truth, res))
