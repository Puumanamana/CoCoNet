import unittest

import os
import sys
sys.path.append(os.path.expanduser("~/CoCoNet"))
from itertools import product

from util import get_kmer_frequency

def slow_kmer_freq(seq,k=4):
    nucl = ["A","C","G","T"]
    kmer_counts = { "".join(nucls): 0 for nucls in product(nucl,repeat=k) }

    for i in range(len(seq)-k+1):
        kmer_counts[seq[i:i+k]] += 1

    return list(kmer_counts.values())


class TestDeepImpute(unittest.TestCase):
    """ """

    def test_kmer_frequency(self, k=4):
        seq = "AAAATCG"
        result = get_kmer_frequency(seq,k)
        truth = slow_kmer_freq(seq,k)

        assert(all(result==truth))
        

if __name__ == "__main__":
    unittest.main()
