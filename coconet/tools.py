"""
Helpful functions for coconet
"""

import os
import logging
from pathlib import Path
from math import ceil
from textwrap import wrap
from functools import lru_cache
from itertools import chain, islice

import h5py
import numpy as np


KMER_CODES = {ord('A'): '00', ord('C'): '01', ord('G'): '10', ord('T'): '11'}
logger = logging.getLogger('<CoCoNet>')

def run_if_not_exists(keys=('output',)):
    """
    Decorator to skip function if the outputs defined in `keys` already exist.
    This decorator is overriden if the environment variable COCONET_CONTINUE
    is not 'Y'.

    Args:
        keys (tuple): outputs to verify
    Returns:
        function
    """

    def run_if_not_exists_key(func):

        def wrapper(*args, **kwargs):
            exists = os.getenv('COCONET_CONTINUE') == 'Y'

            for key in keys:
                if key not in kwargs:
                    exists = False
                    break
                if kwargs[key] is None:
                    exists = False
                    break
                if isinstance(kwargs[key], dict):
                    exists &= all(output.is_file() for output in kwargs[key].values())
                elif isinstance(kwargs[key], list):
                    exists &= all(output.is_file() for output in kwargs[key])
                else:
                    exists &= Path(kwargs[key]).is_file()

            if exists:
                files = ', '.join(map(str, keys))
                msg = f'{func.__name__}: Existing {files} files found. Skipping step'

                if logger is None:
                    print(msg)
                else:
                    logger.info(msg)
                return

            return func(*args, **kwargs)

        return wrapper

    return run_if_not_exists_key

@lru_cache(maxsize=None)
def kmer_count(k, rc=False):
    """
    Counts the number of possible k-mers

    Args:
        k (int): kmer size
        rc (bool): Use canonical k-mers
    Returns:
        int: number of possible kmers
    """
    if not rc:
        return 4**k

    n_palindromes = 0
    if k % 2 == 0:
        n_palindromes = 4**(k//2)
    return (4**k - n_palindromes) // 2 + n_palindromes

@lru_cache(maxsize=None)
def kmer_rc_idx(k=4):
    """
    Get the non redundant kmer indexes when reverse complement is on

    Args:
        k (int): kmer size
    Returns:
        tuple(list, np.array): where the list are the indices to the canonical kmers
          and np.array maps the non-canonical indices to the corresponding canonical index
    """

    mapping = []
    uniq_idx = set()

    for i in range(4**k):
        kmer_rev = ''.join(wrap(f'{4**k-1-i:0{2*k}b}', 2)[::-1])
        i_rev = int(kmer_rev, 2)

        if i_rev not in uniq_idx:
            uniq_idx.add(i)

        if i != i_rev:
            mapping.append([i, i_rev])

    return (list(uniq_idx), np.array(mapping))

def get_kmer_number(sequence, k=4):
    """
    Converts A, C, G, T sequence into sequence of kmer numbers

    Args:
        sequence (str): DNA sequence
        k (int): kmer size
    Returns:
        list of kmer index/hash using the `KMER_CODES` encoding
    """

    kmer_encoding = sequence.translate(KMER_CODES)
    kmer_indices = [int(kmer_encoding[i:i+2*k], 2) for i in range(0, 2*(len(sequence)-k+1), 2)]

    return kmer_indices

def get_kmer_frequency(sequence, kmer=4, rc=False, index=False):
    """
    - Compute kmer occurrences in sequence
    - If kmer index are provided, skip the first part (faster)

    Args:
        sequence (str or int list): DNA sequence
        kmer (int): kmer size for composition
        rc (bool): Use canonical k-mers
        index (bool): whether kmer indexes are provided instead of char sequence

    Returns:
        np.array: `kmer` frequencies for sequence
    """

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

    return occurrences


def get_coverage(pairs, h5_file, window_size, window_step):
    """
    - Extracting coverage from h5 for all pairs of contigs in pairs
    - Smooth coverage with a sliding window of [window_size, window_step]

    Args:
        pairs (np.recarray): Fragment pairs for which to extract coverage values
        h5_file (str): Path to coverage file
        window_size (int): Smoothing window size
        window_step (int): Smoothing window step
    """

    h5data = h5py.File(h5_file, 'r')
    contigs = np.unique(pairs['sp'].flatten())

    try:
        coverage = {ctg: h5data[ctg][:] for ctg in contigs}
    except TypeError as error:
        diff = set(contigs).difference(set(h5data.keys()))
        if diff:
            logger.error((f'{len(diff)} contigs are in the fasta sequences '
                          'but not in the coverage data. '
                          f'For example, {diff.pop()} generates an error.'))
            raise KeyError from error

        logger.error('One contig seem to have a null coverage across all samples')
        raise RuntimeError from error

    h5data.close()

    n_pairs = len(pairs)
    n_samples, _ = np.array(list(coverage.values())[0]).shape
    frag_len = pairs['end'][0, 0] - pairs['start'][0, 0]

    pairs = np.concatenate([pairs[:, 0], pairs[:, 1]])
    sorted_idx = np.argsort(pairs['sp'])

    conv_shape = ceil((frag_len-window_size+1)/window_step)

    coverage_feature = np.zeros([2*n_pairs, n_samples, conv_shape],
                                dtype=np.float32)
    seen = {}

    for i, (sp, start, end) in enumerate(pairs[sorted_idx]):
        cov_sp = seen.get((sp, start))

        if cov_sp is None:
            cov_sp = avg_window(coverage[sp][:, start:end], window_size, window_step, axis=1)
            seen[(sp, start)] = cov_sp

        coverage_feature[sorted_idx[i]] = cov_sp

    return (coverage_feature[:n_pairs, :, :],
            coverage_feature[n_pairs:, :, :])

def avg_window(x, window_size, window_step, axis=1):
    """
    Averaging window with subsampling
    Args:
        x (np.array): Values to process
        window_size (int): Smoothing window size
        window_step (int): Smoothing window step
    Returns:
        np.array
    """
    cumsum = np.cumsum(np.insert(x, 0, 0, axis=axis), axis=axis)

    x_avg = (
        np.take(cumsum, range(window_size, x.shape[axis]+1), axis=axis)
        - np.take(cumsum, range(0, x.shape[axis]+1-window_size), axis=axis)
    ) / float(window_size)

    return np.take(x_avg, range(0, x_avg.shape[axis], window_step), axis=axis)

def chunk(*it, size=2):
    """
    Function adapted from senderle's answer in
    https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks

    Args:
        it (list): list of iterable
        size (int): chunk size
    Returns:
        Iterator over the chunks of size `size` of it
    """

    if not it:
        return []

    it = chain(*it)
    return iter(lambda: tuple(islice(it, size)), ())
