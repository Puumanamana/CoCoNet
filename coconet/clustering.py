"""
Groups all the functions required to run the clustering steps
"""
import logging
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
import torch
import igraph

from coconet.tools import run_if_not_exists, chunk


logger = logging.getLogger('clustering')

@run_if_not_exists()
def make_pregraph(
        model, features, output,
        vote_threshold=None,
        max_neighbors=100,
        buffer_size=50
):
    """
    Pre-cluster contigs into rough communities to be refined later.

    Args:
        model (CompositionModel, CoverageNodel or CoCoNet): PyTorch deep learning model
        features (list): List of binning feature to use (composition, coverage, or both)
        output (str): path to save igraph object
        kw (dict): additional parameters to pass to contig_pair_iterator()
    Returns:
        None
    """


    contigs = features[0].get_contigs()
    
    neighbors = get_neighbors([feature.get_handle('latent') for feature in features])

    # Initialize graph
    graph = igraph.Graph()
    graph.add_vertices(contigs)

    edges = dict()
    
    # Get input data handles
    handles = [(feature.name, feature.get_handle('latent')) for feature in features]

    # Get contig-contig pair generator
    pair_generators = (contig_pair_iterator(ctg, neighb_ctg, graph, max_neighbors=max_neighbors)
                       for (ctg, neighb_ctg) in zip(contigs, neighbors))
    
    edges = compute_pairwise_comparisons(model, handles, pair_generators,
                                         vote_threshold=vote_threshold)
    # Add edges to graph
    if edges:
        graph.add_edges(list(edges.keys()))
        graph.es['weight'] = list(edges.values())

    for handle in handles:
        handle[1].close()

    # Save pre-graph
    graph.write_pickle(output)

@run_if_not_exists(keys=('assignments_file', 'graph_file'))
def refine_clustering(
        model, features, pre_graph_file,
        singletons_file=None, graph_file=None, assignments_file=None,
        theta=0.9, gamma1=0.1, gamma2=0.75, vote_threshold=None,
        **kwargs
):
    """
    Refines graph by computing remaining edges within each cluster
    and re-clustering the whole graph.

    Args:
        model (CompositionModel, CoverageNodel or CoCoNet): PyTorch deep learning model
        features (list): List of binning feature to use (composition, coverage, or both)
        pre_graph_file (str): Path to graph of pre-clustered contigs
        singletons_file (str): Path to singleton contigs that were excluded for the analysis
        assignments_file (str): Path to output bin assignments
        theta (int): binary threshold to draw an edge between contigs
        gamma1 (float): Resolution parameter for leiden clustering in 1st iteration
        gamma2 (float): Resolution parameter for leiden clustering in 2nd iteration
        vote_threshold (float or None): Voting scheme to compare fragments. (Default: disabled)
        kwargs (dict): additional clustering parameters passed to get_communities
    Returns:
        None
    """

    # Data handles
    handles = [(feature.name, feature.get_handle('latent')) for feature in features]
    n_frags = next(iter(handles[0][1].values())).shape[0]

    # Pre-clustering
    pre_graph = igraph.Graph.Read_Pickle(pre_graph_file)
    edge_threshold = theta * n_frags**2

    get_communities(pre_graph, edge_threshold, gamma=gamma1, **kwargs)

    # Refining the clusters
    clusters = np.unique(pre_graph.vs['cluster'])

    logger.info(f'Processing {clusters.size} clusters')

    # Get contig-contig pair generator
    comparisons = []
    for cluster in clusters:
        contigs_c = pre_graph.vs.select(cluster=cluster)['name']
        n_c = len(contigs_c)

        if n_c <= 2:
            continue

        combs = [c for c in combinations(contigs_c, 2) if not pre_graph.are_connected(*c)]
        indices = np.random.choice(len(combs), min(len(combs), int(1e5)))
        comparisons.append([combs[i] for i in indices])

    edges = compute_pairwise_comparisons(model, handles, comparisons)

    # Add edges to graph
    if edges:
        pre_graph.add_edges(list(edges.keys()))
        pre_graph.es['weight'] = list(edges.values())

    for _, handle in handles:
        handle.close()

    get_communities(pre_graph, edge_threshold, gamma=gamma2, **kwargs)
    assignments = pd.Series(dict(zip(pre_graph.vs['name'], pre_graph.vs['cluster'])))

    # Add the rest of the contigs (singletons) set aside at the beginning
    ignored = pd.read_csv(singletons_file, sep='\t', usecols=['contigs'], index_col='contigs')
    ignored['clusters'] = np.arange(len(ignored)) + assignments.values.max() + 1

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


def get_neighbors(handles):
    """
    Returns contigs within a radius in both dimensions for each contig. 
    Ordered by distance in first dimension.
    TODO: use the dimension with the best test scores during learning

    Args:
        handles (tuple list): (feature name, hdf5 file descriptor)
    Returns:
        list: closest {max_neighbors} neighbors of each contig
    """

    n_contigs = len(next(iter(handles[0].keys())))
    
    for i, handle in enumerate(handles):
        components = np.stack([np.array(handle[ctg][:]) for ctg in handle.keys()])
        contig_centers = np.mean(components, axis=1)

        # Distance of each fragment to its center (n_contigs, n_fragments, n_latent)
        distance_to_resp_center = np.sqrt(np.sum(
            (components - contig_centers[:, None, :])**2, axis=2
        ))
        # Radius=mean distance to reference contig + 2 standard deviations
        radii = np.mean(distance_to_resp_center, axis=1) + 2*np.std(distance_to_resp_center, axis=1)
        
        tree = KDTree(contig_centers, leaf_size=min(100, n_contigs))  

        # Preserve the order for the first feature (should be the most predictive feature)
        (new_indices, _) = tree.query_radius(
            contig_centers, radii, sort_results=(i==0), return_distance=True
        )
        
        if i == 0:
            indices = new_indices
        else:
            # Get common neighbors with previous feature(s)
            for i, (idx1, idx2) in enumerate(zip(indices, new_indices)):
                valid = np.isin(idx1, idx2)
                indices[i] = idx1[valid]

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
        model, handles, pairs_generator,
        vote_threshold=None, buffer_size=50
):
    """
    Computes all comparisons between contig pairs produced by `pairs generator` using the provided `model`.
    A given contig-contig comparison involves comparing all n_frag*(n_frag-1)/2 pairs of fragments from
    both contigs. When set, `vote_threshold` imposed a hard threshold on each fragment-fragment comparison
    and converts it to a binary valuye - 1 if P(frag, frag) > vote_threshold and 0 otherwise.
    To save some memory, all comparisons are done in batches and at most `buffer_size`
    contig pairs are compared at once.

    Args:
        model (CompositionModel, CoverageNodel or CoCoNet): PyTorch deep learning model
        handles (list): items are (key, file descriptor) where keys are feature names
          and values are handles to the latent representations of each
          fragment in the contig (shape=(n_fragments, latent_dim))
        pairs_generator (tuple generator): contig pairs to compare
        vote_threshold (float or None): Voting scheme to compare fragments. (None means disabled)
        buffer_size (int): Number of contigs to load at once.
    Returns:
        dict: computed edges with corresponding probability values
    """

    # Get all pairs of fragments between any 2 contigs
    (n_frags, latent_dim) = next(iter(handles[0][1].values())).shape
    n_frag_pairs = n_frags**2
    comb_indices = (np.repeat(np.arange(n_frags), n_frags),
                    np.tile(np.arange(n_frags), n_frags))

    edges = dict()
    
    # Initialize arrays to store inputs of the network
    inputs = [{feature:
               np.zeros((buffer_size*n_frag_pairs, latent_dim), dtype='float32')
               for feature, _ in handles} for _ in range(2)]

    # Smaller chunks
    for i, pairs_buffer in enumerate(chunk(*pairs_generator, size=buffer_size)):
        
        # Load data
        for j, contig_pair in enumerate(pairs_buffer):
            pos = range(j*n_frag_pairs, (j+1)*n_frag_pairs)

            for k, contig in enumerate(contig_pair):
                for (feature, handle) in handles:
                    inputs[k][feature][pos] = handle[contig][:][comb_indices[k]]

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

        if i % 10 == 0 and i > 0:
            logger.info(f'{i*buffer_size:,} contig pairs processed')

    return edges


def get_communities(graph, threshold, gamma=0.5, alg='leiden', n_clusters=None, **kwargs):
    """
    Find communities in `graph` using the `alg` algorithm. Edges with weights lower than `threshold` are
    removed and only edge presence/absence is used for clustering. The other parameters are used
    or not depending on which algorithm is chosen.

    Args:
        graph (igraph.Graph): contig-contig igraph object
        threshold (int): binary edge weight cutoff
        gamma (float): Resolution parameter for cluster density
        alg (str): Community detection algorithm to use
        kwargs (dict): additional parameters passed to the underlying
          community detection algorithm
    Returns:
        None
    """

    bin_graph = graph.copy()
    bin_graph.es.select(weight_lt=threshold).delete()

    if alg == 'leiden':
        communities = bin_graph.community_leiden(
            objective_function="CPM",
            resolution_parameter=gamma,
            n_iterations=-1,
            **kwargs
        )
    elif alg == 'label_propagation':
        communities = bin_graph.community_label_propagation(**kwargs)

    elif alg == 'spectral':
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
