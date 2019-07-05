import pickle
import h5py
import numpy as np
import pandas as pd

from collections import Counter
from clustering import get_neighbors

# with open("neigbors.pkl","rb") as handle:
#     neighbors = pickle.load(handle)

neighbors = get_neighbors('output_data/sim2/representation_cover_nf20.h5')
    
contigs = np.array(list(h5py.File("output_data/sim/representation_cover_nf20.h5").keys()))
viruses = np.array([ vir_name for vir_name,_ in np.char.split(contigs,"_") ])

n_ctg = dict(zip(*np.unique(viruses, return_counts=True)))

infos = pd.DataFrame(pd.Series({ contigs[i]: Counter(viruses[n]) for i,n in enumerate(neighbors)}))
infos["n_ctg"] = [ n_ctg[vir] for vir in viruses]

