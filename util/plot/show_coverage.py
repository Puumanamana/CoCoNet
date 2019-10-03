import os
import sys
import re
from glob import glob

import h5py
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

PARENT_DIR = os.path.join(sys.path[0], '../..')

def plot_coverage(dataset, contigs='auto', n_contigs=4):
    cov_file = glob("{}/input_data/{}/coverage*.h5".format(PARENT_DIR, dataset))[0]
    cov_h5 = h5py.File(cov_file)
    all_contigs = np.fromiter(cov_h5.keys(), dtype='<U128')

    if contigs == ['auto']:
        if cov_h5.get(all_contigs[0]).shape[0] > 5:
            n_contigs = 1
        contigs = np.random.choice(all_contigs, n_contigs)

    dfs = []

    for ctg in contigs:
        df = pd.DataFrame(cov_h5.get(ctg)[:]).T.reset_index()
        df.columns = ["position"]+["sample_{}".format(i) for i in range(df.shape[1]-1)]
        vir_id = re.search(r'\|\d+', ctg)
        if vir_id is None:
            vir_id = ctg
        else:
            vir_id = ctg[:vir_id.end()]
        df["vir_id"] = vir_id
        dfs.append(df)

    dfs = pd.concat(dfs).melt(id_vars=["vir_id", "position"])
    dfs.columns = ["vir_id", "position", "sample", "coverage"]

    if n_contigs == 1:
        g = sns.FacetGrid(data=dfs, col="sample", col_wrap=4, sharey=False, sharex=False)
    else:
        g = sns.FacetGrid(data=dfs, col="sample", row="vir_id", sharey=False, sharex=False)
    g.map(plt.plot, "position", "coverage")
    print(dfs.vir_id.unique())

    plt.show()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Grieg_viral_ocean')
    parser.add_argument('--contigs', type=str, default='auto')
    args = parser.parse_args()

    plot_coverage(args.dataset, contigs=args.contigs.split(' '))
