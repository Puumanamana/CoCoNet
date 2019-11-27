'''
Utils to troubleshoot clustering issues.
Displays adjacency matrix of: 
- Contigs that are clustered when they should not be (wrong)
- Contigs that are not clustered when they should be (fragmented)
'''

import os
import sys
from collections import Counter
import argparse

import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation

PARENT_DIR = os.path.join(sys.path[0], '../..')

sys.path.insert(0, PARENT_DIR)
from experiment import Experiment

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_genomes', type=int, default=2000)
    parser.add_argument('--coverage', type=int, default=10)
    parser.add_argument('--n_samples', type=int, default=15)
    parser.add_argument('--n_iter', type=int, default=0)
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--minsize', type=int, default=2)

    args = parser.parse_args()

    return args

def set_truth(path, assignments):
    truth_file = "{}/truth.csv".format(path)
    if not os.path.exists(truth_file):
        mask = assignments.contigs.str.contains('|', regex=False)
        assignments = assignments.loc[mask]
        assignments.truth = pd.factorize(assignments.contigs.str.split('|').str[0])[0]
    else:
        truth = pd.read_csv(truth_file, index_col=0, header=None, names=['contigs', 'cluster'])
        mask = np.isin(assignments.contigs, truth.index)
        assignments = assignments.loc[mask]
        assignments.truth = truth.loc[assignments.contigs, 'cluster'].values
    return assignments, mask

def load_and_process_assignments(cfg, minsize=0):

    assignments = pd.read_csv(cfg.outputs['clustering']['refined_assignments'])
    assignments.columns = ['contigs', 'clusters', 'truth']
    assignments, mask = set_truth(cfg.outdir, assignments)
    assignments.index = np.arange(assignments.shape[0])
    assignments.index.name = 'ctg_id'

    grouped = {'by_clusters': (assignments.reset_index()
                               .groupby("clusters")
                               .filter(lambda x: len(x) > minsize)
                               .groupby("clusters")),
               'by_truth': (assignments.reset_index()
                            .groupby("truth")
                            .filter(lambda x: len(x) > minsize)
                            .groupby("truth"))}

    return assignments, grouped, mask.values

def load_and_process_adj(cfg, truth, mask):
    '''
    Preprocess adjacency matrix:
    - Replace -1 with NaN
    - Wrong hits set to negative
    '''

    adj_mat = np.load(cfg.outputs['clustering']['refined_adjacency_matrix'])[:, mask][mask, :]
    adj_mat[adj_mat == -1] = np.nan

    not_together = truth.reshape(-1, 1) != truth.reshape(1, -1)
    adj_mat[not_together] *= -1

    return adj_mat

def sort_error_types(grouped):

    # Fragmented bins
    incomplete = grouped['by_truth'].clusters.agg([count_sorted, count_uniques])
    incomplete = incomplete.loc[incomplete.count_uniques > 1, 'count_sorted']

    # Wrong bins (grouping different viruses)
    heterogeneous = grouped['by_clusters'].truth.agg([count_sorted, count_uniques])
    heterogeneous = heterogeneous.count_sorted[heterogeneous.count_uniques > 5]

    return {'FN': incomplete,
            'FP': heterogeneous}

def count_sorted(l):
    '''
    Count the occurrences of elements in l and sort them
    '''

    counts = Counter(l).items()
    return sorted(counts, key=lambda x: x[1], reverse=True)

def count_uniques(l):
    '''
    Count nuber of uniques elements in l
    '''

    return len(set(l))

def display_matches(adj_mat, subset, assgn):
    '''
    Display adjacency matrix for the indices in subset
    Performs clustering first to group the rows/columns
    '''

    adjacency_sub = adj_mat[subset, :][:, subset]

    adj_sub_no_na = adjacency_sub.copy()
    adj_sub_no_na[np.isnan(adj_sub_no_na)] = 0
    ctg_order = np.argsort(AffinityPropagation().fit(adj_sub_no_na).labels_)

    print(assgn['clusters'].values[ctg_order])
    print(assgn['contigs'].values[ctg_order])

    cmap = matplotlib.cm.get_cmap('RdYlGn')
    cmap.set_bad(color='grey')
    plt.matshow(adjacency_sub[ctg_order, :][:, ctg_order], cmap=cmap, vmin=-900, vmax=900)

    plt.show()

def main():

    args = parse_args()

    if args.name != '':
        name = args.name
    else:
        name = "{}_{}_{}_{}".format(args.n_genomes, args.coverage, args.n_samples, args.n_iter)

    config = Experiment(name, root_dir=PARENT_DIR)

    assignments, grouped, mask = load_and_process_assignments(config, minsize=args.minsize)
    adjacency_matrix = load_and_process_adj(config, assignments.truth.values, mask)

    errors_df = sort_error_types(grouped)

    while 1:
        prompt = input("FN/FP/[q]/i: ").strip().upper()

        if prompt == 'FN':
            # Random fragmented virus
            rd_virus = errors_df['FN'].sample(1).index[0]
            # Indices of the contigs the virus is split
            indices = grouped['by_truth'].agg(list).loc[rd_virus, "ctg_id"]
        elif prompt == 'FP':
            # Random wrong bin
            rd_bin = errors_df['FP'].sample(1).index[0]
            indices = grouped['by_clusters'].agg(list).loc[rd_bin, "ctg_id"]
        elif prompt == 'I':
            import ipdb
            ipdb.set_trace()
        else:
            break

        display_matches(adjacency_matrix, indices, assignments.loc[indices])

if __name__ == '__main__':
    main()
