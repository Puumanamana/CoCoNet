import pandas as pd
import numpy as np
import h5py
import seaborn as sns
import matplotlib.pyplot as plt
# import networkx as nx

def plot_subheatmap(adj,mapping,n_viruses=20):
    selection = np.random.choice(mapping.virus.unique(),n_viruses)
    selection_idx = mapping.reset_index().set_index('virus').loc[selection,'index'].values
    adj_sub = adj[selection_idx,:][:,selection_idx]
    adj_sub[adj_sub<0] = -900

    sns.heatmap(adj_sub,cmap='RdYlBu')
    plt.show()

if __name__ == '__main__':
    
    import sys
    dataset = sys.argv[1]

    root_dir = "/home/cedric/projects/viral_binning/CoCoNet/output_data/{}".format(dataset)
    adjacency_matrix = np.load("{}/adjacency_matrix_nf30.npy".format(root_dir))
    contigs = h5py.File("{}/representation_cover_nf30.h5".format(root_dir)).keys()

    sep = '_'*('sim' in dataset) + '|'*('split' in dataset)

    mapping = pd.DataFrame([ [ ctg, ctg.split(sep)[0] ] for ctg in contigs ], columns=['contig','virus'])
    
        
    plot_subheatmap(adjacency_matrix,mapping)
