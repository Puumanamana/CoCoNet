import os
import numpy as np
import pandas as pd
import h5py

import torch
from sklearn.metrics.pairwise import euclidean_distances
import networkx as nx
import community

from Bio import SeqIO
from progressbar import progressbar

from util import get_kmer_frequency, avg_window

def save_repr_all(model,n_frags,frag_len,kmer_list,rc,window_size,
                  fasta=None,coverage_h5=None,outputs=None):

    cov_h5 = h5py.File(coverage_h5,'r')

    repr_h5 = { key: h5py.File(filename,'w') for key,filename in outputs.items() }    

    for contig in progressbar(SeqIO.parse(fasta,"fasta"), max_value=len(cov_h5)):

        step = int((len(contig)-frag_len) / n_frags)

        fragment_boundaries = [ (step*i, step*i+frag_len)
                            for i in range(n_frags) ]

        # contig_kmers = np.array(get_kmer_number(str(contig.seq),k=kmer))[fragment_slices]
        
        x_composition = torch.from_numpy(np.stack(
            [ get_kmer_frequency(str(contig.seq)[start:stop],
                                 kmer_list=kmer_list,rc=rc)
              for (start,stop) in fragment_boundaries]
        ).astype(np.float32)) # Shape = (n_frags, 4**k)

        fragment_slices = np.array([np.arange(start,stop)
                                    for (start,stop) in fragment_boundaries ])
        coverage_genome = np.array(cov_h5.get(contig.id)[:]).astype(np.float32)[:,fragment_slices]
        
        coverage_genome = np.swapaxes(coverage_genome,1,0)
        
        x_coverage = torch.from_numpy(
            np.apply_along_axis(
                lambda x: avg_window(x,window_size), 2, coverage_genome
            ).astype(np.float32) )
        
        x_repr = model.compute_repr(x_composition,x_coverage)
        
        [ handle.create_dataset(contig.id, data=x_repr[key].detach().numpy(), dtype=np.float32)
          for key, handle in repr_h5.items() ]
        
    [ handle.close() for handle in repr_h5.values() ]
        
def get_neighbors(file_h5):
    print("Calculating neighbors")
    handle = h5py.File(file_h5,'r')

    # data.shape = ( n_contigs, n_frags, latent_dim )
    data = np.stack([np.array(handle.get(ctg)[:])
                     for ctg in handle.keys()])

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

def cluster(model,repr_h5,outputdir,max_neighbor=50,prob_thresh=0.75,n_frags=50,hits_threshold=0.95):

    adj_mat_file = "{}/adjacency_matrix_nf{}.npy".format(outputdir,n_frags)
    handles = {key: h5py.File(filename) for key,filename in repr_h5.items() }

    contigs = np.array(list(handles['coverage'].keys()))

    if not os.path.exists(adj_mat_file):
        neighbors = get_neighbors(repr_h5['coverage'])
        
        for i,ni in enumerate(get_neighbors(repr_h5['composition'])):
            neighbors[i] = np.intersect1d(neighbors[i],ni)
        
        combination_idx = [
            np.repeat(np.arange(n_frags),n_frags),
            np.tile(np.arange(n_frags),n_frags)
        ]
        
        adjacency_matrix = np.identity(len(contigs)) * n_frags**2
        adjacency_matrix[adjacency_matrix==0] = -1
        adjacency_matrix = adjacency_matrix.astype(int)

        for k, ctg in progressbar(enumerate(contigs),max_value=len(contigs)):

            x_ref = { key: torch.from_numpy(np.array(handle.get(ctg)[:])[combination_idx[0]])
                      for key,handle in handles.items() }

            # Discard neighbors that we already calculated
            neighbors_k = neighbors[k]
            scores = adjacency_matrix[k,neighbors_k]
            new_neighbors_k = neighbors_k[scores < 0][:max_neighbor]

            for ni in new_neighbors_k:
                x_other = { key: torch.from_numpy(np.array(
                    handle.get(contigs[ni])[:]
                )[combination_idx[1]])
                            for key,handle in handles.items() }
                probs = model.combine_repr(x_ref,x_other).detach().numpy()
                
                adjacency_matrix[k,ni] = sum(probs>prob_thresh)
                adjacency_matrix[ni,k] = adjacency_matrix[k,ni]
                
            # matches_k = neighbors_k[adjacency_matrix[k,neighbors_k] > hits_threshold * n_frags**2]
            # if len(matches_k) > 1:
            #     print("Matches for contig {}: {}"
            #           .format(contigs[k],"-".join(contigs[matches_k]))
            #     )
                    
        np.save(adj_mat_file,adjacency_matrix)

    else:
        adjacency_matrix = np.load(adj_mat_file)

    # Remove -1 (NAs) from the matrix
    adjacency_matrix[adjacency_matrix < 0] = 0
    
    threshold = hits_threshold * n_frags**2
    G = nx.from_numpy_matrix((adjacency_matrix>threshold).astype(int))

    assignments = pd.DataFrame({'clusters': list(community.best_partition(G).values()),
                                'contigs': contigs,
                                'truth': [x.split('|')[0] for x in contigs]})

    assignments.to_csv("{}/assignments_nf{}.csv".format(outputdir,n_frags))

    clusters_grouped = assignments.groupby('clusters')['truth'].agg([lambda x: len(set(x)),len]).sort_values(by='len')
    clusters_grouped.columns = ["purity","csize"]

    print(clusters_grouped)
    
    # print(clusters_grouped[ (clusters_grouped.purity == 1) & (clusters_grouped.csize>1) ].shape)



    
