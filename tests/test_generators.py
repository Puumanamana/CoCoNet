import unittest

import os
import sys
sys.path.append(os.path.expanduser("~/CoCoNet"))
from itertools import product

import numpy as np
import pandas as pd
import h5py

from util import get_kmer_frequency, get_coverage, avg_window

def slow_kmer_freq(seq,k=4):
    
    nucl = ["A","C","G","T"]
    kmer_counts = { "".join(nucls): 0 for nucls in product(nucl,repeat=k) }

    for i in range(len(seq)-k+1):
        kmer_counts[seq[i:i+k]] += 1

    return list(kmer_counts.values())

def slow_coverage(pairs,h5file,window_size):
    
    def smooth(x):
        return avg_window(x,window_size)
    
    h5data = h5py.File(h5file)

    X1,X2 = [],[]
    for sp1,s1,e1,sp2,s2,e2 in pairs.values:
        
        cov1 = h5data.get(sp1)[:,s1:e1]
        X1.append(np.apply_along_axis( smooth,1,cov1 ))
        
        cov2 = h5data.get(sp2)[:,s2:e2]
        X2.append(np.apply_along_axis( smooth,1,cov2 ))

    return np.array(X1,dtype=np.float32),np.array(X2,dtype=np.float32)

class TestGenerators(unittest.TestCase):
    """ """

    def test_get_kmer_frequency(self, k=4):
        
        seq = "AAAATCG"
        result = get_kmer_frequency(seq,k)
        truth = slow_kmer_freq(seq,k)

        assert(all(result==truth))

    def test_get_coverage(self,window_size=3):
        
        pairs = pd.DataFrame([["V0_0",0,50,"V0_0",100,150],
                              ["V1_0",0,50,"V0_0",50,100]],
                             columns=pd.MultiIndex.from_product([['A','B'], ['sp','start','end']]))

        X1,X2 = get_coverage(pairs,'/home/cedric/CoCoNet/tests/test.h5',window_size)
        T1,T2 = slow_coverage(pairs,'/home/cedric/CoCoNet/tests/test.h5',window_size)

        assert (np.sum(X1!=T1) + np.sum(X2!=T2) == 0)

if __name__ == "__main__":
    unittest.main()
