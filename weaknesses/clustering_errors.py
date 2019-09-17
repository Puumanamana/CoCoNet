from collections import Counter

import pandas as pd
import numpy as np
import h5py

import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation

def preproc_adj(adj_mat, contig_names):
    adj_mat[adj_mat == -1] = np.nan

    truth = pd.factorize([x.split('|')[0] for x in contig_names])[0]
    not_together = np.mod(np.sqrt(np.matmul(truth.reshape(-1, 1), truth.reshape(1, -1))), 1) != 0
    adj_mat[not_together] *= -1

    return adj_mat

def count_sorted(L):
    counts = Counter(L).items()
    return sorted(counts, key=lambda x: x[1], reverse=True)

def count_uniques(L):
    return len(set(L))

def display_matches(adj_mat, subset, contig_names):
    adjacency_sub = adj_mat[subset, :][:, subset]

    # Display matches in adjacency table
    print(adjacency_sub)
    print(contig_names[subset])

    adj_sub_no_na = adjacency_sub.copy()
    adj_sub_no_na[np.isnan(adj_sub_no_na)] = 0
    ctg_order = np.argsort(AffinityPropagation().fit(adj_sub_no_na).labels_)

    cmap = matplotlib.cm.get_cmap('RdYlGn')
    cmap.set_bad(color='grey')
    plt.matshow(adjacency_sub[ctg_order, :][:, ctg_order], cmap=cmap)

    plt.show()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='.')
    parser.add_argument('--minsize', type=int, default=3)
    args = parser.parse_args()

    ctgs = np.fromiter(h5py.File("{}/representation_cover_nf30.h5".format(args.path)).keys(),
                       dtype='<U128')
    ctgs_w_truth = np.core.defchararray.find(ctgs,'|') != -1
    ctgs = ctgs[ctgs_w_truth]

    assignments = pd.read_csv("{}/leiden_nf30.csv"
                              .format(args.path)).loc[ctgs_w_truth]
    assignments['ctg_id'] = np.arange(assignments.shape[0])
    adjacency_matrix = np.load("{}/adjacency_matrix_nf30.npy"
                               .format(args.path))[ctgs_w_truth, :][:, ctgs_w_truth]
    adjacency_matrix = preproc_adj(adjacency_matrix, ctgs)

    grouped_by_clusters = (assignments
                           .groupby("clusters")
                           .filter(lambda x: len(x) > args.minsize)
                           .groupby("clusters"))

    grouped_by_truth = (assignments
                        .groupby("truth")
                        .filter(lambda x: len(x) > args.minsize)
                        .groupby("truth"))

    # Fragmented bins
    fragmentation = grouped_by_truth.clusters.agg([count_sorted, count_uniques])
    fragmentation = fragmentation.loc[fragmentation.count_uniques > 1, 'count_sorted']

    # Wrong bins (grouping different viruses)
    wrong = grouped_by_clusters.truth.agg([count_sorted, count_uniques])
    wrong = wrong.count_sorted[wrong.count_uniques > 5]

    while 1:
        prompt = input("Fragmentation/Wrong/Stop (f/w/s): ").strip()

        if prompt == 'f':
            # Random fragmented virus
            rd_virus = fragmentation.sample(1).index[0]
            # Indices of the contigs the virus is split
            indices = grouped_by_truth.agg(list).loc[rd_virus, "ctg_id"]
        elif prompt == 'w':
            # Random wrong bin
            rd_bin = wrong.sample(1).index[0]
            indices = grouped_by_clusters.agg(list).loc[rd_bin, "ctg_id"]
        else:
            break

        display_matches(adjacency_matrix, indices, ctgs)
