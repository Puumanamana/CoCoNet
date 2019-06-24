import h5py
from glob import glob

import pandas as pd
import numpy as np

from progressbar import progressbar

print("WARNING: V4863 has been manually removed. See issue CAMISIM issue #60")

metadata = pd.read_csv("metadata.csv",index_col="accession")
metadata.V_id = metadata.V_id.str.split("_").str.get(0)
mapping = (metadata
           .groupby("V_id")
           .agg({"start": list,
                 "end": list})
)

genomes_length = metadata.groupby("V_id")["end"].agg("last")

def convert(h5data,virus,depth_dir="camisim/coverage/*"):
    datasets = [ pd.read_csv("{}/{}.txt".format(sample,virus),
                             header=None, sep="\t", names=["acc","pos","depth"])
                 for sample in sorted(glob(depth_dir)) ]

    coverage = np.zeros([genomes_length[virus],len(datasets)],dtype=np.uint32)

    for i,dataset in enumerate(datasets):
        if dataset.size > 0:
            coverage[dataset.pos.values-1,i] = dataset.depth.values

    contigs = mapping.loc[virus]

    for i,(start,end) in enumerate(zip(contigs.start,contigs.end)):
        h5data.create_dataset("{}_{}".format(virus,i),
                              data=coverage[start:end,:],
                              dtype='uint32')
    
h5data = h5py.File("coverage.h5")
[ convert(h5data,virus) for virus in progressbar(genomes_length.drop("V4863").index) ]

