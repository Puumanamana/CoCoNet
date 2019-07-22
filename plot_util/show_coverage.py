import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
import h5py

inputdir = os.path.expanduser("~/projects/viral_binning/CoCoNet/input_data")

def plot_coverage(dataset, contigs=None, n_contigs=4):
    cov_h5 = h5py.File("{}/{}/coverage_contigs.h5".format(inputdir,dataset))

    if contigs is None:
        contigs = np.random.choice(cov_h5,n_contigs)
    
    dfs = []

    for ctg in contigs:
        df = pd.DataFrame(cov_h5.get(ctg)[:]).T.reset_index()
        df.columns = ["position"]+["sample_{}".format(i) for i in range(df.shape[1]-1)]  
        df["V_id"] = ctg[:re.search('_\d+',ctg).end()]
        dfs.append(df)

    dfs = pd.concat(dfs).melt(id_vars=["V_id","position"])
    dfs.columns = ["V_id","position","sample","coverage"]

    g = sns.FacetGrid(data=dfs, col="sample", row="V_id", sharey=False, sharex=False)
    g.map(plt.plot,"position","coverage")
    print(dfs.V_id.unique())
    
    plt.show()

if __name__ == '__main__':
    import sys

    dataset = sys.argv[1]

    if len(sys.argv) > 2:
        plot_coverage(dataset, contigs=sys.argv[2:])
    else:
        plot_coverage(dataset)

    
    
