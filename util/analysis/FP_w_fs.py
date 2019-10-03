import h5py
import numpy as np
import pandas as pd
from Bio import SeqIO
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '../..')
from experiment import Experiment

cfg = Experiment('5_5', root_dir='../..')

contigs = np.fromiter(h5py.File(cfg.outputs['repr']['composition']).keys(), dtype='<U128')
ctg_len = pd.Series({seq.id: np.log2(len(seq.seq)).astype(int)
                     for seq in SeqIO.parse(cfg.inputs['filtered']['fasta'], 'fasta') })
ctg_len = ctg_len.loc[contigs].reset_index()[0]
ctg_len -= ctg_len.min()

adj_mat = np.load(cfg.outputs['clustering']['refined_adjacency_matrix'])

truth = {i: ctg.split('|') for i, ctg in enumerate(contigs)}

nb_bins = len(ctg_len.unique())

FP_res = np.zeros((nb_bins, nb_bins))
nb_comp = np.zeros((nb_bins, nb_bins))

for i in range(len(contigs)):
    print("{:,}/{:,}".format(i, len(contigs)), end='\r')
    idx_hits = np.where(adj_mat[i] > -1)[0]
    l1 = ctg_len[i]
    for k in idx_hits:
        if truth[i] != truth[k]:
            l2 = ctg_len[k]
            FP_res[l1, l2] += adj_mat[i, k]
            FP_res[l2, l1] += adj_mat[i, k]
            nb_comp[l1, l2] += 1
            nb_comp[l2, l1] += 1

nb_comp[nb_comp == 0] = 1
FP_res = FP_res/nb_comp

plt.matshow(FP_res, cmap='Greens')
plt.xlabel('Length (log2 bp)')
plt.ylabel('Length (log2 bp)')
plt.show()
