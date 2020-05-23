'''
Helpful routines for coconet
- Decorator to prevent re-executing finished steps (for resuming)
- kmer calculation
- coverage extraction from a hdf5 and npy file
- smoothing window
'''

import sys
from pathlib import Path
from math import ceil
from textwrap import wrap
from functools import lru_cache

import h5py
import numpy as np

KMER_CODES = {ord('A'): '00', ord('C'): '01', ord('G'): '10', ord('T'): '11'}

def run_if_not_exists(keys=('output',)):
    '''
    Decorator to skip function if output exists
    '''

    def run_if_not_exists_key(func):

        def wrapper(*args, **kwargs):
            exists = True

            for key in keys:
                if key not in kwargs:
                    exists = False
                    break
                if kwargs[key] is None:
                    exists = False
                    break
                if isinstance(kwargs[key], dict):
                    exists &= all(output.is_file() for output in kwargs[key].values())
                else:
                    exists &= Path(kwargs[key]).is_file()

            if exists:
                files = [str(kwargs[key]) for key in keys]
                print('{} already exist. Skipping step'.format(files))
                return

            return func(*args, **kwargs)

        return wrapper

    return run_if_not_exists_key

@lru_cache(maxsize=None)
def kmer_count(k, rc=False):
    if not rc:
        return 4**k

    n_palindromes = 0
    if k % 2 == 0:
        n_palindromes = 4**(k//2)
    return (4**k - n_palindromes) // 2 + n_palindromes

@lru_cache(maxsize=None)
def kmer_rc_idx(k=4):
    '''
    Get the non redundant kmer indexes when reverse complements is on
    '''

    mapping = []
    uniq_idx = set()

    for i in range(4**k):
        kmer_rev = ''.join(wrap('{:08b}'.format(4**k-1-i), 2)[::-1])
        i_rev = int(kmer_rev, 2)

        if i_rev not in uniq_idx:
            uniq_idx.add(i)

        if i != i_rev:
            mapping.append([i, i_rev])

    return (list(uniq_idx), np.array(mapping))

def get_kmer_number(sequence, k=4):
    '''
    Converts A, C, G, T sequence into sequence of kmer numbers
    '''

    kmer_encoding = sequence.translate(KMER_CODES)
    kmer_indices = [int(kmer_encoding[i:i+2*k], 2) for i in range(0, 2*(len(sequence)-k+1), 2)]

    return kmer_indices

def get_kmer_frequency(sequence, kmer=4, rc=False, index=False, norm=False):
    '''
    - Compute kmer occurrences in sequence
    - If kmer index are provided, skip the first part (faster)
    '''

    if rc:
        uniq_idx, rev_mapping = kmer_rc_idx(kmer)
    if not index:
        kmer_indices = get_kmer_number(sequence, kmer)
    else:
        kmer_indices = sequence

    occurrences = np.bincount(kmer_indices, minlength=4**kmer)

    if rc:
        occurrences[rev_mapping[:, 0]] += occurrences[rev_mapping[:, 1]]
        occurrences = occurrences[uniq_idx]
    if norm:
        occurrences = occurrences / np.sum(occurrences)

    return occurrences

def get_coverage(pairs, coverage_h5, window_size, window_step, pbar=None):
    '''
    - Extracting coverage from h5 for all pairs of contigs in pairs
    - Smooth coverage with a sliding window of [window_size, window_step]
    '''

    h5data = h5py.File(coverage_h5, 'r')
    contigs = np.unique(pairs['sp'].flatten())

    try:
        coverage_data = {ctg: np.array(h5data.get(ctg)[:]) for ctg in contigs}
    except TypeError:
        diff = set(contigs).difference(set(h5data.keys()))
        if diff:
            print('''
            Error: {} contigs are in the fasta sequences but not in the coverage.
            For example, {} generates an error.
            '''.format(len(diff), diff[0]))
        else:
            sys.exit('One contig seem to have a null coverage across all samples')

    n_pairs = len(pairs)
    n_samples, _ = np.array(list(coverage_data.values())[0]).shape
    frag_len = pairs['end'][0, 0] - pairs['start'][0, 0]

    pairs = np.concatenate([pairs[:, 0], pairs[:, 1]])
    sorted_idx = np.argsort(pairs['sp'])
    
    conv_shape = ceil((frag_len-window_size+1)/window_step)

    coverage_feature = np.zeros([2*n_pairs, n_samples, conv_shape],
                                dtype=np.float32)
    seen = {}

    for i, (species, start, end) in enumerate(pairs[sorted_idx]):
        try:
            cov_sp = seen.get((species, start), None)

            if cov_sp is None:
                cov_sp = np.apply_along_axis(
                    lambda x: avg_window(x, window_size, window_step),
                    1,
                    coverage_data[species][:, start:end]
                )
                seen[(species, start)] = cov_sp

            coverage_feature[sorted_idx[i]] = cov_sp
        except:
            import ipdb;ipdb.set_trace()
            
        if pbar is not None:
            pbar.update(0.5)

    return (coverage_feature[:n_pairs, :, :],
            coverage_feature[n_pairs:, :, :])

def avg_window(x, window_size, window_step):
    x_conv = np.convolve(x, np.ones(window_size)/window_size, mode="valid")
    return x_conv[::window_step]
