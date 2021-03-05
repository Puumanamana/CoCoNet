"""
Groups all the functions required to run the clustering steps
"""

import logging
from itertools import combinations

import numpy as np
import pandas as pd
import scipy
import h5py
import sklearn.neighbors
import hnswlib
import torch
import igraph
from coconet.tools import run_if_not_exists, chunk


logger = logging.getLogger('<clustering>')

@run_if_not_exists()
def make_pregraph(
        model, latent_vectors, output,
        vote_threshold=None,
        max_neighbors=100,
        buffer_size=500,
        threads=1
):
    """
    Pre-cluster contigs into rough communities to be refined later.

    Args:
        model (CompositionModel, CoverageNodel or CoCoNet): PyTorch deep learning model
        latent_vectors (tuple list): items are (feature name, dict) where dict keys are contigs,
          and values are the latent representations of the fragments in the contig
        output (str): path to save igraph object
        vote_threshold (float): hard threshold for voting scheme. If set to none, the
          probabilities are summed
        max_neighbors (int): maximum number of neighbor contigs to compare each query contig with
        buffer_size (int): maximum number of contigs to compute in one batch. Consider lowering
          this value if you have limited RAM.
        threads (int): number of thread to use
    Returns:
        None
    """

    contigs = list(latent_vectors[0][1].keys())

    neighbors = get_neighbors(latent_vectors, threads=threads)

    # Initialize graph
    graph = igraph.Graph()
    graph.add_vertices(contigs)
    graph.es['weight'] = []

    # Get contig-contig pair generator
    pairs_generators = (contig_pair_iterator(ctg, neighb_ctg, graph, max_neighbors=max_neighbors)
                       for (ctg, neighb_ctg) in zip(contigs, neighbors))

    edges = compute_pairwise_comparisons(
        model, latent_vectors, pairs_generators, vote_threshold=vote_threshold,
        buffer_size=buffer_size
    )

    # Add edges to graph
    if edges:
        graph.add_edges(list(edges.keys()))
        graph.es['weight'] = list(edges.values())

    # Save pre-graph
    graph.write_pickle(output)

@run_if_not_exists(keys=('assignments_file', 'graph_file'))
def refine_clustering(
        model, latent_vectors, pre_graph_file,
        graph_file=None, assignments_file=None, dtr_file=None,
        theta=0.9, gamma1=0.1, gamma2=0.75, vote_threshold=None,
        buffer_size=500, **kwargs
):
    """
    Refines graph by computing remaining edges within each cluster
    and re-clustering the whole graph.

    Args:
        model (CompositionModel, CoverageNodel or CoCoNet): PyTorch deep learning model
        latent_vectors (tuple list): items are (feature name, dict) where dict keys are contigs,
          and values are the latent representations of the fragments in the contig
        pre_graph_file (str): Path to graph of pre-clustered contigs
        graph_file (str): Path to output graph
        assignments_file (str): Path to output bin assignments
        dtr_file (str): path to DTR contig list (to include as singleton bins)
        theta (int): binary threshold to draw an edge between contigs
        gamma1 (float): Resolution parameter for leiden clustering in 1st iteration
        gamma2 (float): Resolution parameter for leiden clustering in 2nd iteration
        vote_threshold (float or None): Voting scheme to compare fragments. (Default: disabled)
        buffer_size (int): maximum number of contigs to compute in one batch. Consider lowering
          this value if you have limited RAM.
        kwargs (dict): additional clustering parameters passed to get_communities
    Returns:
        None
    """

    # Get dimensions from any latent vectors
    (n_frags, _) = next(iter(latent_vectors[0][1].values())).shape

    # Pre-clustering
    graph = igraph.Graph.Read_Pickle(pre_graph_file)
    edge_threshold = theta * n_frags**2

    get_communities(graph, edge_threshold, gamma=gamma1, **kwargs)

    # Refining the clusters
    clusters = np.unique(graph.vs['cluster'])

    logger.info(f'Processing {clusters.size} clusters')

    # Get contig-contig pair generator
    comparisons = []
    for cluster in clusters:
        contigs_c = graph.vs.select(cluster=cluster)['name']
        n_c = len(contigs_c)

        if n_c <= 2:
            continue

        combs = [c for c in combinations(contigs_c, 2) if not graph.are_connected(*c)]
        indices = np.random.choice(len(combs), min(len(combs), int(1e5)))
        comparisons.append([combs[i] for i in indices])

    edges = compute_pairwise_comparisons(
        model, latent_vectors, comparisons, vote_threshold=vote_threshold,
        buffer_size=buffer_size
    )

    # Add edges to graph
    if edges:
        graph.add_edges(list(edges.keys()))
        graph.es['weight'] = list(edges.values())

    # Add complete genomes
    if dtr_file is not None and dtr_file.is_file():
        dtr_contigs = set(ctg.split('\t')[0].strip() for ctg in open(dtr_file))
        cur_assignments = graph.es['cluster']
        graph.add_vertices(dtr_contigs)
        cl_max = max(cur_assignments)
        graph.es['cluster'] = cur_assignments + [cl_max+1+i for (i, _) in enumerate(dtr_contigs)]

    get_communities(graph, edge_threshold, gamma=gamma2, **kwargs)
    graph.write_pickle(graph_file)

    # Write the cluster in .csv format
    communities = pd.Series(graph.vs['cluster'], index=graph.vs['name'])
    communities.to_csv(assignments_file, header=False)

def get_neighbors(latent_vectors, threads=1):
    """
    Returns contigs within a radius in both dimensions for each contig.
    Ordered by distance in first dimension.

    Args:
        latent_vectors (tuple list): items are (feature name, dict) where dict keys are contigs,
          and values are the latent representations of the fragments in the contig
        threads (int): Number of threads to use
    Returns:
        int list: closest {max_neighbors} neighbors of each contig
    """

    n_contigs = len(latent_vectors[0][1])

    for i, (_, data) in enumerate(latent_vectors):
        components = np.stack(list(data.values()))
        contig_centers = np.mean(components, axis=1)

        # Distance of each fragment to its center (n_contigs, n_fragments, n_latent)
        distance_to_resp_center = np.sqrt(np.sum(
            (components - contig_centers[:, None, :])**2, axis=2
        ))
        radius = np.percentile(distance_to_resp_center, 90)

        if n_contigs < 20000:
            tree = sklearn.neighbors.NearestNeighbors(
                radius=radius, algorithm='ball_tree', n_jobs=threads
            )
            tree.fit(contig_centers)

            # Preserve the order for the first feature (should be the most predictive feature)
            (_, new_indices) = tree.radius_neighbors(
                contig_centers, sort_results=(i==0), return_distance=True
            )
        else: # use a more efficient library: hnswlib
            p = hnswlib.Index(space='l2', dim=components.shape[-1])
            p.init_index(max_elements=n_contigs, ef_construction=200, M=32)
            p.set_num_threads(threads)
            p.add_items(contig_centers)
            new_indices, distances = p.knn_query(contig_centers, k=min(1000, n_contigs))
            new_indices = [idx[dist <= radius**2] for (idx, dist) in zip(new_indices, distances)]
        if i == 0:
            indices = new_indices
        else:
            # Get common neighbors with previous feature(s)
            for j, (idx1, idx2) in enumerate(zip(indices, new_indices)):
                valid = np.isin(idx1, idx2)
                indices[j] = idx1[valid]

    return indices

def contig_pair_iterator(
        contig, neighbors_index=None, graph=None, max_neighbors=100
):
    """
    Generator of (contig, contig1) pairs defined where `contig1`
    are in the `neighbors` of `contig` that were not already edges in the `graph`

    Args:
        contig (str): reference contig to be compared with the neighbors
        neighbors_index (list): neighboring contigs index to consider
        graph (igraph.Graph): Graph with already computed edges
        max_neighbors (int): Maximum number of neighbors for a given contig
    Returns:
        Generator: Generates (contig, contigs) pairs
    """

    contigs = np.array(graph.vs['name'])
    # Since we use .pop(), we need the last neighbors to be the closest
    neighbors_index = list(neighbors_index[::-1])

    i = 0
    while neighbors_index:
        neighbor_index = neighbors_index.pop()

        neighbor = contigs[neighbor_index]

        if i < max_neighbors:
            if graph.are_connected(contig, neighbor) or contig == neighbor:
                continue
            i += 1

            yield (contig, neighbor)


def compute_pairwise_comparisons(
        model, latent_vectors, pairs_generator,
        vote_threshold=None, buffer_size=500
):
    """
    Computes all comparisons between contig pairs produced by `pairs generator`
    using the provided `model`. A given contig-contig comparison involves
    comparing all n_frag*(n_frag-1)/2 pairs of fragments from both contigs.
    When set, `vote_threshold` imposed a hard threshold on each fragment-fragment
    comparison and converts it to a binary valuye - 1 if P(frag, frag) > vote_threshold
    and 0 otherwise. To save some memory, all comparisons are done in batches
    and at most `buffer_size` contig pairs are compared at once.

    Args:
        model (CompositionModel, CoverageNodel or CoCoNet): PyTorch deep learning model
        latent_vectors (list): items are (feature name, dict) where dict is the latent
          representations for all fragments in the contig (shape=(n_fragments, latent_dim))
        pairs_generator (tuple generator): contig pairs to compare
        vote_threshold (float or None): Voting scheme to compare fragments. (None means disabled)
        buffer_size (int): Number of contigs to load at once.
    Returns:
        dict: computed edges with corresponding probability values
    """

    # Get dimensions from any latent vectors
    (n_frags, latent_dim) = next(iter(latent_vectors[0][1].values())).shape

    n_frag_pairs = n_frags**2
    comb_indices = (np.repeat(np.arange(n_frags), n_frags),
                    np.tile(np.arange(n_frags), n_frags))

    edges = dict()

    # Initialize arrays to store inputs of the network
    inputs = [{feature:
               np.zeros((buffer_size*n_frag_pairs, latent_dim), dtype='float32')
               for feature, _ in latent_vectors} for _ in range(2)]

    # Smaller chunks
    for i, pairs_buffer in enumerate(chunk(*pairs_generator, size=buffer_size)):

        # Load data
        for (feature, data) in latent_vectors:
            for j, contig_pair in enumerate(pairs_buffer):
                pos = range(j*n_frag_pairs, (j+1)*n_frag_pairs)

                for k, contig in enumerate(contig_pair):
                    inputs[k][feature][pos] = data[contig][comb_indices[k]]

        # Convert to pytorch
        inputs_torch = [
            {feature: torch.from_numpy(matrix)
             for feature, matrix in input_j.items()}
            for input_j in inputs
        ]

        if len(inputs_torch[0]) == 1: # Only one feature type
            feature = next(iter(inputs_torch[0].keys()))
            inputs_torch = [x[feature] for x in inputs_torch]

        # make prediction
        probs = model.combine_repr(*inputs_torch).detach().numpy()[:, 0]

        if vote_threshold is not None:
            probs = probs > vote_threshold

        # Save edge weight
        for j, contig_pair in enumerate(pairs_buffer):
            edges[contig_pair] =  sum(probs[j*n_frags**2:(j+1)*n_frags**2])

        if i % 100 == 0 and i > 0:
            logger.info(f'{i*buffer_size:,} contig pairs processed')

    return edges


def get_communities(graph, threshold, gamma=0.5, algorithm='leiden', n_clusters=None, **kwargs):
    """
    Find communities in `graph` using the `algorithm` algorithm. Edges with weights lower
    than `threshold` are removed and only edge presence/absence is used for clustering.
    The other parameters are used or not depending on which algorithm is chosen.

    Args:
        graph (igraph.Graph): contig-contig igraph object
        threshold (int): binary edge weight cutoff
        gamma (float): Resolution parameter for cluster density
        algorithm (str): Community detection algorithm to use
        n_clusters (int): If spectral clustering is used, maximum number of cluster to allow
        kwargs (dict): additional parameters passed to the underlying
          community detection algorithm
    Returns:
        None
    """

    bin_graph = graph.copy()
    bin_graph.es.select(weight_lt=threshold).delete()

    if algorithm == 'leiden':
        communities = bin_graph.community_leiden(
            objective_function="CPM",
            resolution_parameter=gamma,
            n_iterations=-1,
            **kwargs
        )
    elif algorithm == 'spectral':
        if 'cluster' in bin_graph.vs.attribute_names():
            # Pre-clustering
            n_clusters = int(n_clusters*gamma)

        if not isinstance(n_clusters, int):
            logger.critical('n_clusters is required for spectral clustering')
            raise ValueError
        communities = bin_graph.community_leading_eigenvector(int(n_clusters * gamma))

    communities = pd.Series(dict(
        enumerate(communities)
    )).explode()

    # Set the "bin" attribute
    graph.vs['cluster'] = communities.sort_values().index


def salvage_contigs(bins, coverage, min_bin_size=3, output='recruits.csv'):
    """
    Recruits contigs shorter than 2048 in the existing bins.
    Write the new assignments in a separate file.
    Args:
        bins (str): Path to output bin assignments
        coverage (str): Path to h5 coverage
    Returns:
        None
    """

    handle = h5py.File(coverage, 'r')

    min_bin_size = 3
    n_samples = next(iter(handle.values())).shape[0]
    percentiles = np.arange(0, 100, 1)

    assignments = pd.read_csv(bins, header=None, names=['contigs', 'bins'])
    assignments = assignments.groupby('bins').filter(lambda x: len(x) >= min_bin_size)
    contigs_per_bin = assignments.groupby('bins').contigs.agg(list)

    queries_cov = [(ctg, np.percentile(handle[ctg][:], percentiles, axis=1).T)
                   for ctg in handle.keys()
                   if 1024 < handle[ctg].shape[1] < 2048]

    recruits = dict()
    scores = dict()

    for bin_id, contigs in contigs_per_bin.iteritems():
        if len(contigs) < min_bin_size:
            continue

        logger.info(f'Checking bin {bin_id}')

        # Raw coverage values for bin
        rcov_bin = [[np.percentile(handle[ctg][s, :], percentiles) for ctg in contigs]
                    for s in range(n_samples)]
        # print(np.stack([np.mean(x, axis=1) for x in rcov_bin]))
        intra_dist = np.zeros(n_samples)

        # Compute intra-bin similarities
        for i, ref in enumerate(rcov_bin):
            intra_dist[i] = np.nanmean(scipy.spatial.distance.pdist(ref, metric='euclidean'))

        # Compute bin-query similarities
        for (query_id, query_cov) in queries_cov:
            score_query = 0
            for s in range(n_samples):
                dist = np.nanmean(scipy.spatial.distance.cdist(
                    query_cov[s][None, :], rcov_bin[s],
                    metric='euclidean'
                ))

                score_query += dist

                if dist > intra_dist[s]:
                    break

            else:
                score_query /= n_samples
                prev_score = scores.get(query_id, np.inf)

                if score_query < prev_score:
                    recruits[query_id] = bin_id
                    scores[query_id] = score_query

    if recruits:
        recruits = pd.Series(recruits).rename('bin_id')
        recruits.index.name = 'contig'
        recruits.to_csv(output, index=True, header=True, sep='\t')
    else:
        logger.info('No contig <2048bp recruited in bins')
