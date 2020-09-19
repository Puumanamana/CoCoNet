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
                                 n_frags=30, bin_id=-1,
                                 buffer_size=50):
    '''
    Compare each contig in [contigs] with its [neighbors]
    by running the neural network for each fragment combinations
    between the contig pair
    '''

    n_frag_pairs = n_frags**2
    comb_indices = (np.repeat(np.arange(n_frags), n_frags),
                    np.tile(np.arange(n_frags), n_frags))

    contigs = np.array(graph.vs['name'])
    if bin_id > 0:
        contigs = contigs[neighbors]
    
    contigs_chunks = (contigs[i:i+buffer_size]
                      for i in range(0, len(contigs), buffer_size))

    if len(contigs) > 20:
        logger.info(f"Processing cluster ({len(contigs)} contigs)")

    # Get latent variable dimension
    latent_dim = []
    for (feature, handle) in handles:
        a_value = list(handle.values())[0]
        latent_dim.append((feature, a_value.shape))

    edges = {}
    # Case 1 (bin_id == -1): we are computing the pre-graph
    # Case 2 (else): we are computed edges in sub-graph
    for i, chunk in enumerate(contigs_chunks):
        if bin_id == -1:
            if i*buffer_size % max(1, len(contigs)//100) == 0:
                logger.debug(f"{i*buffer_size:,} contigs done")

        # 1) get list of (contig, other)
        ctg_pairs = []
        for j, ctg in enumerate(chunk):
            neighb_idx = neighbors[i*buffer_size+j] if bin_id == -1 else neighbors
            processed = np.isin(neighb_idx, graph.neighbors(ctg))
            neighb_j = contigs[neighb_idx][~processed][:int(max_neighbors)]

            ctg_pairs += [(ctg, other) for other in neighb_j
                          if other != ctg and (other, ctg) not in edges]

        if not ctg_pairs:
            continue

        # 2) create empty numpy array to hold data
        x = [{feature: np.zeros((len(ctg_pairs)*n_frag_pairs, dim[1]),
                                dtype=np.float32)
              for (feature, dim) in latent_dim}
             for _ in range(2)]
        
        # 3) fill array and cache already loaded contigs
        loaded = {}
        for k, pair in enumerate(ctg_pairs):
            indices = {feature: np.arange(k*n_frag_pairs, (k+1)*n_frag_pairs)
                       for feature in x[0]}
            
            for (l, ctg) in enumerate(pair):
                if not ctg in loaded:
                    loaded[ctg] = {feature: handle[ctg][:][comb_indices[l]]
                                   for (feature, handle) in handles}

                for (feature, data) in loaded[ctg].items():
                    x[l][feature][indices[feature]] = data

        for x_i in x:
            for feature, data in x_i.items():
                x_i[feature] = torch.from_numpy(data)

            if len(x_i) == 1: # Only one feature type
                x_i = x_i[feature]

        # 4) convert to torch and make prediction
        probs = model.combine_repr(*x).detach().numpy()[:, 0]

        if vote_threshold is not None:
            probs = probs > vote_threshold

        for i, pair in enumerate(ctg_pairs):
            edges[pair] = sum(probs[i*n_frag_pairs:(i+1)*n_frag_pairs])

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

        if i > 0 and i % (len(clusters)//5) == 0:
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
