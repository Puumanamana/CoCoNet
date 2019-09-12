import os
from itertools import combinations
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

from util import get_kmer_frequency, avg_window

def save_repr_all(model,config):

    cov_h5 = h5py.File(config.inputs['filtered']['coverage_h5'],'r')

    repr_h5 = { key: h5py.File(filename,'w') for key,filename in config.outputs['repr'].items() }

    n_frags = config.clustering['n_frags']

    for contig in progressbar(SeqIO.parse(config.inputs['filtered']['fasta'],"fasta"),
                              max_value=len(cov_h5)):
        step = int((len(contig)-config.fl) / n_frags)

        fragment_boundaries = [ (step*i, step*i+config.fl) for i in range(n_frags) ]

        x_composition = torch.from_numpy(np.stack(
            [ get_kmer_frequency(str(contig.seq)[start:stop],
                                 kmer_list=config.kmer_list,rc=config.rc)
              for (start,stop) in fragment_boundaries]
        ).astype(np.float32)) # Shape = (n_frags, 4**k)

        fragment_slices = np.array([np.arange(start,stop)
                                    for (start,stop) in fragment_boundaries ])
        coverage_genome = np.array(cov_h5.get(contig.id)[:]).astype(np.float32)[:,fragment_slices]        
        coverage_genome = np.swapaxes(coverage_genome,1,0)
        
        x_coverage = torch.from_numpy(
            np.apply_along_axis(
                lambda x: avg_window(x,config.wsize,config.wstep), 2, coverage_genome
            ).astype(np.float32) )
        
        x_repr = model.compute_repr(x_composition,x_coverage)
        
        [ handle.create_dataset(contig.id, data=x_repr[key].detach().numpy(), dtype=np.float32)
          for key, handle in repr_h5.items() ]
        
    [ handle.close() for handle in repr_h5.values() ]
        
def get_neighbors(file_h5):
    print("Calculating neighbors")
    handle = h5py.File(file_h5,'r')

    # data.shape = ( n_contigs, n_frags, latent_dim )
    data = np.stack([np.array(handle.get(ctg)[:]) for ctg in handle.keys()])
    # center of each contigs (n_contigs, latent_dim)
    contig_centers = np.mean(data,axis=1)
    # pairwise distances between contig centers
    distances_to_other = euclidean_distances(contig_centers)
    # distance of each fragment to its center
    distance_to_resp_center = np.sqrt(np.sum(
        (data - contig_centers[:,None,:])**2,axis=2
        ))
    # radius of each contig (median distance from fragment to center
    radii = np.median(distance_to_resp_center, axis=1)
    # Condition: neighbors need to be within radii units from the center
    within_range = distances_to_other < radii.reshape(-1,1)

    # Get neighbor indices
    indices = np.arange(len(radii))
    valid_indices = [ indices[wr] for wr in within_range ]
    sorted_indices = [ np.array(sorted(indices, key=lambda idx: distances_to_other[i,idx]))
                       for i,indices in enumerate(valid_indices) ]

    return sorted_indices

def cluster(model,config):

    handles = {key: h5py.File(filename) for key,filename in config.outputs['repr'].items() }    

    contigs = np.array(list(handles['coverage'].keys()))
    n_frags = config.clustering['n_frags']
    
    if not os.path.exists(config.outputs['clustering']['adjacency_matrix']):
        neighbors = get_neighbors(config.outputs['repr']['coverage'])
        
        for i,ni in enumerate(get_neighbors(config.outputs['repr']['composition'])):
            neighbors[i] = np.intersect1d(neighbors[i],ni)
        
        ref_idx, other_idx = ( np.repeat(np.arange(n_frags),n_frags),
                               np.tile(np.arange(n_frags),n_frags) )

        adjacency_matrix = (1 + np.identity(len(contigs))*n_frags**2) - 1

        for k, ctg in progressbar(enumerate(contigs),max_value=len(contigs)):

            x_ref = { key: torch.from_numpy(np.array(handle.get(ctg)[:])[ref_idx])
                      for key,handle in handles.items() }

            # Discard neighbors that we already calculated
            scores = adjacency_matrix[k,neighbors[k]]
            new_neighbors_k = neighbors[k][scores < 0][:config.clustering['max_neighbors']]

            for ni in new_neighbors_k:
                x_other = { key: torch.from_numpy(np.array( handle.get(contigs[ni])[:] )[other_idx])
                            for key,handle in handles.items() }
                probs = model.combine_repr(x_ref,x_other).detach().numpy()
                # Get number of expected matches
                adjacency_matrix[k,ni] = sum(probs) # sum(probs>prob_thresh)
                adjacency_matrix[ni,k] = adjacency_matrix[k,ni]
                                    
        np.save(config.outputs['clustering']['adjacency_matrix'],adjacency_matrix)

    else:
        adjacency_matrix = np.load(config.outputs['clustering']['adjacency_matrix'])

    # Remove -1 (=NAs) from the matrix
    adjacency_matrix[adjacency_matrix < 0] = 0
    
    threshold = config.clustering['hits_threshold'] * n_frags**2

    for algo in config.clustering['algo'].split(','):
        if algo == 'louvain':
            G = nx.from_numpy_matrix((adjacency_matrix>threshold).astype(int))
            communities = list(community.best_partition(G).values())
        elif algo == 'leiden':
            G = igraph.Graph.Adjacency((adjacency_matrix>threshold).tolist())
            leiden_generator = enumerate(leidenalg.find_partition(G, leidenalg.ModularityVertexPartition))
            communities = (pd.Series(dict(leiden_generator))
                           .explode()
                           .sort_values()
                           .index)

        sep = "_"
        if "|" in "".join(contigs):
            sep = "|"

        assignments = pd.DataFrame({'clusters': communities,
                                    'contigs': contigs,
                                    'truth': [x.split(sep)[0] for x in contigs]})
        assignments.clusters = pd.factorize(assignments.clusters)[0]
        assignments.truth = pd.factorize(assignments.truth)[0]

        outputname = config.outputs['clustering']['assignments'].replace('leiden',algo),replace('louvain',algo)
        assignments.to_csv(outputname)

def refine_clusters(assignments,model,config):
    handles = {key: h5py.File(filename) for key,filename in config.outputs['repr'].items() }

    contigs = np.array(list(handles['coverage'].keys()))
    
    clusters = assignments.reset_index().groupby('clusters',as_index=False).agg(list)[['clusters','index']]

    for c, indices in clusters.values:
        indices1,indices2 = tuple(zip(*[ (i,j) for i,j in combinations(indices,2) if adj[i,j]>=0 ]))

        X1 = { key: torch.from_numpy(np.array(
            [ handle.get(contigs[k])[:] for k in indices1 ]
        )) for key,handle in handles.items() }
        
        X2 = { key: torch.from_numpy(np.array(
            [ handle.get(contigs[k])[:] for k in indices2 ]
        )) for key,handle in handles.items() }
        
        probs = model.combine_repr(X1,X2).detach().numpy()
        # Get number of expected matches
        
        adj[indices1,indices2] = sum(probs)
        adj[indices2,indices1] = sum(probs)        
    
