'''
Tests for data generators
'''

from itertools import product
from textwrap import wrap
import pytest

import numpy as np
import h5py

from coconet.tools import get_kmer_frequency, get_coverage, avg_window
from .data import generate_pair_file, generate_h5_file

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

def slow_coverage(pairs, h5file, window_size, window_step):

    def smooth(x):
        return avg_window(x, window_size, window_step)

    h5data = h5py.File(h5file, 'r')

    X1, X2 = [], []
    for (sp1, st1, end1), (sp2, st2, end2) in pairs:

        cov1 = h5data.get(sp1)[:, st1:end1]
        X1.append(np.apply_along_axis(smooth, 1, cov1))

        cov2 = h5data.get(sp2)[:, st2:end2]
        X2.append(np.apply_along_axis(smooth, 1, cov2))

    return np.array(X1, dtype=np.float32), np.array(X2, dtype=np.float32)


def test_get_kmer_frequency(k=4):

    seq = "AAAATCG"
    result = get_kmer_frequency(seq, k)
    truth = slow_kmer_freq(seq, k)

    assert all(result == truth)

def test_get_kmer_frequency_with_rc(k=4):

    seq = "AAAAT"
    result = get_kmer_frequency(seq, k, rc=True)
    truth = slow_kmer_freq(seq, k, rc=True)

    assert all(result == truth)

def test_smoothing(wsize=3, wstep=2):
    x_in = np.array([1, 2, 3, 4, 5])
    result = avg_window(x_in, wsize, wstep)

    assert np.mean(result - np.array([2, 4])) < 1e-10

def test_get_coverage(window_size=4):

    pairs = generate_pair_file(save=False)
    data_h5 = generate_h5_file(30, 40)

    (X1, X2) = get_coverage(pairs, data_h5, window_size, window_size // 2)
    (T1, T2) = slow_coverage(pairs, data_h5, window_size, window_size // 2)

    data_h5.unlink()

    assert np.sum(X1 != T1) + np.sum(X2 != T2) == 0

def test_get_coverage_with_unmatched_ctg(window_size=4):

    pairs = generate_pair_file(save=False)
    data_h5 = generate_h5_file(30)

    with pytest.raises(TypeError):
        assert get_coverage(pairs, data_h5, window_size, window_size // 2)

    data_h5.unlink()
