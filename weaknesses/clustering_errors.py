import sys

import pandas as pd
import numpy as np

from collections import Counter
import matplotlib.pyplot as plt

if len(sys.argv) > 1:
    input_dir = sys.argv[1]
else:
    input_dir = "../output_data/sim"

adj = np.load("{}/adjacency_matrix_nf50.npy".format(input_dir))
assignments = pd.read_csv("{}/assignments_nf50.csv".format(input_dir),
                          index_col=0)

def count_sorted(L):
    counts = Counter(L).items()
    return sorted(counts, key=lambda x:x[1], reverse=True)

def uniques(L):
    return len(set(L))

clusters = assignments.reset_index().groupby("clusters").agg(list)
true_clusters = assignments.reset_index().groupby("truth").agg(list)

# Fragmented bins
fragmentation = assignments.groupby("truth")["clusters"].agg([count_sorted,uniques])
fragmentation = fragmentation.count_sorted[fragmentation['uniques']>1]

# Wrong bins (grouping different viruses)
wrong = assignments.groupby("clusters")["truth"].agg([count_sorted,uniques])
wrong = wrong.count_sorted[wrong['uniques']>1]

def rd_wrong_display():
    # Random wrong bin
    rd_bin = wrong.sample(1).index[0]

    indices = clusters.loc[rd_bin,"index"]

    # Display corresponding contigs
    print(assignments.contigs[indices].tolist())
    # Display matches in adjacency table
    print(adj[indices,:][:,indices])

    plt.matshow(adj[indices,:][:,indices])
    plt.show()

def rd_frag_display():
    # Random fragmented virus
    rd_virus = fragmentation.sample(1).index[0]
    # Indices of the contigs  the virus is split
    indices = true_clusters.loc[rd_virus,"index"]

    # Display corresponding contigs
    print(assignments.contigs[indices].tolist())
    # Display matches in adjacency table
    print(adj[indices,:][:,indices])

    plt.matshow(adj[indices,:][:,indices])
    plt.show()

another = True
while another:
    prompt = input("Fragmentation/Wrong/Stop (f/w/s): ").strip()

    if prompt == 'f':
        rd_frag_display()
    elif prompt == 'w':
        rd_wrong_display()
    else:
        another = False
