import unittest

import os
import sys
from itertools import product
from textwrap import wrap

import numpy as np
import pandas as pd
import h5py

PARENT_DIR = os.path.join(sys.path[0], '..')
sys.path.insert(0, PARENT_DIR)

from tools import get_kmer_frequency, get_coverage, avg_window

def get_rc_indices(k):
    uniq_idx = set()

    for i in range(4**k):
        kmer_rev = ''.join(wrap('{:08b}'.format(4**k-1-i), 2)[::-1])
        idx = int(kmer_rev, 2)

        if idx not in uniq_idx:
            uniq_idx.add(i)
    return list(uniq_idx)

def slow_kmer_freq(seq, k=4, rc=False):

    rc_trans = str.maketrans('ACGT', 'TGCA')
    nucl = ["A", "C", "G", "T"]
    kmer_counts = {"".join(nucls): 0 for nucls in product(nucl, repeat=k)}

    for i in range(len(seq)-k+1):
        kmer_counts[seq[i:i+k]] += 1

        if rc:
            seq_rc = seq[i:i+k].translate(rc_trans)[::-1]
            if seq_rc != seq[i:i+k]:
                kmer_counts[seq_rc] += 1

    freqs = np.array(list(kmer_counts.values()))

    if rc:
        indices = get_rc_indices(k)
        freqs = freqs[indices]

    return freqs

def slow_coverage(pairs, h5file, window_size):

    def smooth(x):
        return avg_window(x, window_size)

    h5data = h5py.File(h5file)

    X1, X2 = [], []
    for sp1, s1, e1, sp2, s2, e2 in pairs.values:

        cov1 = h5data.get(sp1)[:, s1:e1]
        X1.append(np.apply_along_axis(smooth, 1, cov1))

        cov2 = h5data.get(sp2)[:, s2:e2]
        X2.append(np.apply_along_axis(smooth, 1, cov2))

    return np.array(X1, dtype=np.float32), np.array(X2, dtype=np.float32)

class TestGenerators(unittest.TestCase):

    def test_get_kmer_frequency(self, k=4):

        seq = "AAAATCG"
        result = get_kmer_frequency(seq, k)
        truth = slow_kmer_freq(seq, k)

        assert all(result == truth)

    def test_get_kmer_frequency_with_rc(self, k=4):

        seq = "AAAAT"
        result = get_kmer_frequency(seq, k, rc=True)
        truth = slow_kmer_freq(seq, k, rc=True)

        assert all(result == truth)

    def test_get_coverage(self, window_size=3):

        pairs = pd.DataFrame([["V0_0", 0, 50, "V0_0", 100, 150],
                              ["V1_0", 0, 50, "V0_0", 50, 100]],
                             columns=pd.MultiIndex.from_product([['A', 'B'], ['sp', 'start', 'end']]))

        X1, X2 = get_coverage(pairs, '/home/cedric/CoCoNet/tests/test.h5', window_size)
        T1, T2 = slow_coverage(pairs, '/home/cedric/CoCoNet/tests/test.h5', window_size)

        assert np.sum(X1 != T1) + np.sum(X2 != T2) == 0

if __name__ == "__main__":
    unittest.main()
