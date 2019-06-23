import os
import sys
import h5py

import pandas as pd
import numpy as np

metadata = pd.read_csv("../metadata.csv",index_col="accession")
metadata.V_id = metadata.V_id.str.split("_").str.get(0)
mapping = metadata.groupby(level=0)["V_id"].agg("first")

genomes_length = metadata.groupby("V_id")["end"].agg("last")

genome = os.environ["VIRUS"]
sample = os.environ["SAMPLE"]
coverage = np.zeros(genomes_length[genome],dtype=np.uint32)

handle = h5py.File('coverages.hdf5', 'a')

for i,line in enumerate(sys.stdin):
    genome, pos, count = line.split("\t")
    coverage[int(pos)-1] = int(count)

handle.create_dataset("sample_{}/{}".format(sample,genome), data=coverage)
