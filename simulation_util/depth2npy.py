import pickle
from glob import glob

import pandas as pd
import numpy as np

from progressbar import progressbar

metadata = pd.read_csv("metadata.csv",index_col="accession")
metadata.V_id = metadata.V_id.str.split("_").str.get(0)
mapping = metadata.groupby(level=0)["V_id"].agg("first")

genomes_length = metadata.groupby("V_id")["end"].agg("last")

def convert(virus,depth_dir="camisim/coverage/*"):
    datasets = [ pd.read_csv("{}/{}.txt".format(sample,virus),
                             header=None, sep="\t", names=["acc","pos","depth"])
                 for sample in sorted(glob(depth_dir)) ]
    
    coverage = np.zeros([genomes_length[virus],len(datasets)],dtype=np.uint32)

    for i,dataset in enumerate(datasets):
        if dataset.size > 0:
            coverage[dataset.pos.values-1,i] = dataset.depth.values

    return coverage

coverages = { virus: convert(virus) for virus in progressbar(genomes_length.index) }

with open("camisim/coverage_all.pickle","wb") as handle:
    pickle.dump(handle)


