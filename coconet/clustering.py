'''
Groups all the functions required to run the clustering steps
'''
import logging
from itertools import combinations

import numpy as np
import pandas as pd

import torch
import igraph
import leidenalg

from coconet.tools import run_if_not_exists, chunk

logger = logging.getLogger('clustering')


@run_if_not_exists()
def make_pregraph(
        model, features, output,
        vote_threshold=None,
        max_neighbors=100,
        buffer_size=50
):
    '''
    Args:
        model: PyTorch neural network model
        features: List of binning feature to use (composition, coverage, or both)
        output: path to save igraph object
        kw: dict of additional parameters to pass to contig_pair_iterator()
    Returns:
        None
    '''

    contigs = features[0].get_contigs()
    
    # Intersect neighbors for all features
    neighbors_each = [feature.get_neighbors_index() for feature in features]

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
        theta=0.9, gamma1=0.1, gamma2=0.75, vote_threshold=None
):
    '''
    Args:
        model: PyTorch neural network model
        features: List of binning feature to use (composition, coverage, or both)
        pre_graph_file: Path to graph of pre-clustered contigs
        singletons_file: Path to singleton contigs that were excluded for the analysis
        assignments_file: Path to output bin assignments
        theta: binary threshold to draw an edge between contigs
        gamma1: Resolution parameter for leiden clustering in 1st iteration
        gamma2: Resolution parameter for leiden clustering in 2nd iteration
        vote_threshold: Voting scheme to compare fragments. (Default: disabled)
    Returns:
        None
    '''

    # Data handles
    handles = [(feature.name, feature.get_handle('latent')) for feature in features]
    n_frags = next(iter(handles[0][1].values())).shape[0]

    # Pre-clustering
    pre_graph = igraph.Graph.Read_Pickle(pre_graph_file)
    edge_threshold = theta * n_frags**2

    get_communities(pre_graph, edge_threshold, gamma=gamma1)

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
    
    # get_communities(pre_graph, edge_threshold, gamma=gamma2)
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


   
def contig_pair_iterator(
        contig, neighbors_index=None, graph=None, max_neighbors=100
):
    '''
    Args:
        contig: reference contig to be compared with the neighbors
        neighbors_index: neighboring contigs index to consider
        graph: Graph with already computed edges
        max_neighbors: Maximum number of neighbors for a given contig
    Returns:
        Generator of (contig, contigs) pairs
    '''
    
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
    '''
    Args:
        model: PyTorch neural network model
        handles: (key, descriptor) list where keys are feature names 
          and values are handles to the latent representations of each 
          fragment in the contig (shape=(n_fragments, latent_dim))
        pairs_generator: (ctg1, ctg2) generator corresponding to the comparisons to compute
        vote_threshold: Voting scheme to compare fragments. (Default: disabled)
        buffer_size: Number of contigs to load at once.
    Returns:
        dictionnary of edges computed with corresponding probability values
    '''

    # Get all pairs of fragments between any 2 contigs
    (n_frags, latent_dim) = next(iter(handles[0][1].values())).shape
    n_frag_pairs = n_frags**2
    comb_indices = (np.repeat(np.arange(n_frags), n_frags),
                    np.tile(np.arange(n_frags), n_frags))

    edges = dict()

    # Smaller chunks
    for i, pairs_buffer in enumerate(chunk(*pairs_generator, size=buffer_size)):
        # Initialize arrays to store inputs of the network
        inputs = [{feature:
            np.zeros((buffer_size*n_frag_pairs, latent_dim), dtype='float32')
                   for feature, _ in handles} for _ in range(2)]

        # Load data
        for j, contig_pair in enumerate(pairs_buffer):
            pos = range(j*n_frag_pairs, (j+1)*n_frag_pairs)
        
            for k, contig in enumerate(contig_pair):
                for (feature, handle) in handles:
                    inputs[k][feature][pos] = handle[contig][:][comb_indices[k]]

        # Convert to pytorch
        for j, input_j in enumerate(inputs):
            for feature, matrix in input_j.items():
                inputs[j][feature] = torch.from_numpy(matrix)

            if len(input_j) == 1: # Only one feature type
                input_j = input_j[feature]
  
        # make prediction
        probs = model.combine_repr(*inputs).detach().numpy()[:, 0]

        if vote_threshold is not None:
            probs = probs > vote_threshold

        # Save edge weight
        for j, contig_pair in enumerate(pairs_buffer):
            edges[contig_pair] =  sum(probs[j*n_frags**2:(j+1)*n_frags**2])

        if i % 10 == 0:
            logger.info(f'{i*buffer_size:,} contig pairs processed')

    return edges

    
def get_communities(graph, threshold, gamma=0.5):
    '''
    Args:
        graph: contig-contig igraph object
        threshold: binary edge weight cutoff
        gamma: Resolution parameter for leiden clustering
    Returns:
        None
    '''

    cluster_graph = graph.copy()
    cluster_graph.es.select(weight_lt=threshold).delete()

    optimiser = leidenalg.Optimiser()

    partition = leidenalg.CPMVertexPartition(cluster_graph, resolution_parameter=gamma)
    optimiser.optimise_partition(partition)

    communities = pd.Series(dict(enumerate(partition))).explode()

    # communities = pd.Series(dict(enumerate(
    #     cluster_graph.community_leading_eigenvector(
    #         clusters=int(189*gamma)
    #     )
    # )))

    # Set the "cluster" attribute
    graph.vs['cluster'] = communities.sort_values().index

