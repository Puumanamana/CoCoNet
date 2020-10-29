'''
ts for data generators
'''

from itertools import product
import pytest

import numpy as np
import h5py

from coconet.tools import get_kmer_frequency, get_coverage, avg_window, kmer_rc_idx
from .data import generate_pair_file, generate_h5_file

def slow_kmer_freq(seq, k=4, rc=False):

    nucls = "ACGT"
    rc_trans = str.maketrans(nucls, nucls[::-1])

    kmer_counts = {"".join(kmer): 0 for kmer in product(list(nucls), repeat=k)}

    for i in range(len(seq)-k+1):
        kmer_counts[seq[i:i+k]] += 1

        if rc:
            seq_rc = seq[i:i+k].translate(rc_trans)[::-1]
            if seq_rc != seq[i:i+k]:
                kmer_counts[seq_rc] += 1

    freqs = np.array(list(kmer_counts.values()))

    if rc:
        indices, _ = kmer_rc_idx(k)
        freqs = freqs[indices]

    return freqs

def slow_coverage(pairs, h5file, window_size, window_step):

    def smooth(x):
        return avg_window(x, window_size, window_step, 0)

    h5data = h5py.File(h5file, 'r')

    X1, X2 = [], []
    for (sp1, st1, end1), (sp2, st2, end2) in pairs:

        cov1 = h5data.get(sp1)[:, st1:end1]
        X1.append(np.apply_along_axis(smooth, 1, cov1))

        cov2 = h5data.get(sp2)[:, st2:end2]
        X2.append(np.apply_along_axis(smooth, 1, cov2))

    return np.array(X1, dtype=np.float32), np.array(X2, dtype=np.float32)


def test_get_kmer_frequency(k=4, n_repeat=10):

    is_equal = True
    random_seqs = [''.join(letters) for letters in np.random.choice(list('ACGT'), [20, n_repeat])]

    for seq in random_seqs:
        result = get_kmer_frequency(seq, k)
        truth = slow_kmer_freq(seq, k)

        is_equal &= all(result == truth)

    assert is_equal

def test_get_kmer_frequency_with_rc(k=4, n_repeat=10):

    is_equal = True
    random_seqs = [''.join(letters) for letters in np.random.choice(list('ACGT'), [20, n_repeat])]

    for seq in random_seqs:
        result = get_kmer_frequency(seq, k, rc=True)
        truth = slow_kmer_freq(seq, k, rc=True)

        is_equal &= all(result == truth)

    assert is_equal

def test_smoothing(wsize=3, wstep=2):
    x_in = np.array([1, 2, 3, 4, 5])
    result = avg_window(x_in, wsize, wstep, 0)

    assert np.mean(result - np.array([2, 4])) < 1e-10

def test_get_coverage(window_size=4):

    pairs = generate_pair_file(save=False)
    data_h5 = generate_h5_file(30, 40, filename='coverage.h5')

    (X1, X2) = get_coverage(pairs, data_h5, window_size, window_size // 2)
    (T1, T2) = slow_coverage(pairs, data_h5, window_size, window_size // 2)

    data_h5.unlink()

    assert np.sum(X1 != T1) + np.sum(X2 != T2) == 0

def test_get_coverage_with_unmatched_ctg(window_size=4):

    pairs = generate_pair_file(save=False)
    data_h5 = generate_h5_file(30, filename='coverage.h5')

    with pytest.raises(KeyError):
        assert get_coverage(pairs, data_h5, window_size, window_size // 2)

    data_h5.unlink()


if __name__ == '__main__':
    test_get_coverage()
