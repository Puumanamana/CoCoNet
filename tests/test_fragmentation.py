import unittest

import os
import sys
sys.path.append(os.path.expanduser("~/CoCoNet"))
from itertools import combinations

import numpy as np
import pandas as pd
from Bio import SeqIO

from fragmentation import calculate_optimal_dist
from fragmentation import make_negative_pairs, make_positive_pairs, make_pairs

def get_pairs_frag_with_dist(n_frags,dist):
    count = 0
    for i,j in combinations(range(n_frags),2):
        if abs(i-j) >= dist:
            count += 1
    return count
    

class TestFragmentation(unittest.TestCase):
    """ """

    def test_optimal_dist(self, n_frags=50, fppc=5):
        
        dist = calculate_optimal_dist(n_frags,fppc)
        
        total_pairs_opt = get_pairs_frag_with_dist(n_frags,dist)
        total_pairs_more = get_pairs_frag_with_dist(n_frags,dist+1)

        is_opt = ( (total_pairs_more < fppc) # With a bigger distance, we have less than fppc fragments
                   & (total_pairs_opt >= fppc) )

        print("{} < {} < {}".format(total_pairs_more, fppc, total_pairs_opt))

        assert is_opt

    def test_negative_pairs(self, n_examples=10):

        fragments = pd.Series({'A': 200, 'B': 50, 'C': 100})
        result = make_negative_pairs(fragments, n_examples, 3)

        assert sum(result.A.sp != result.B.sp) == n_examples
        assert result.shape == (n_examples,6)

    def test_positive_pairs(self,fppc=30,contig_frags=100):
        
        min_dist = calculate_optimal_dist(contig_frags,fppc)
        result = make_positive_pairs('V0_0',4,contig_frags,fppc)

        assert sum(result.A.sp == result.B.sp) == fppc
        assert sum(np.abs(result.A.start.astype(int)
                      - result.B.start.astype(int)) < min_dist) == 0
        assert result.shape == (fppc,6)

    def test_all_pairs(self,n_examples=1000):
        contigs = SeqIO.parse("tests/test_genomes.fasta","fasta")

        step = 2
        frag_len = 5
        
        result = make_pairs(contigs,step,frag_len,n_examples=n_examples)

        assert abs(np.mean(result.A.sp == result.B.sp) -0.5) < 0.1
        assert result.shape == (2*n_examples, 6)
        

if __name__ == "__main__":
    unittest.main()
