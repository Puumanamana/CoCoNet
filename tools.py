from typing import Dict
from pathlib import Path
from math import ceil
from time import time
from textwrap import wrap
from functools import lru_cache

import h5py
import numpy as np

KMER_CODES = {ord('A'): '00', ord('C'): '01', ord('G'): '10', ord('T'): '11'}
KMER_CODES_REV = {ord('A'): '11', ord('C'): '10', ord('G'): '01', ord('T'): '00'}

def timer(func):
    def wrapper(*args, **kwargs):
        t_init = time()
        res = func(*args, **kwargs)
        duration = time() - t_init

        print("{}: {} s".format(func.__name__, duration))
        return res

    return wrapper


def run_if_not_exists(func):
    def wrapper(*args, **kwargs):
        if isinstance(kwargs['output'], dict):
            exists = all(output.is_file() for output in kwargs['output'].values())
        else:
            exists = Path(kwargs['output']).is_file()
        if exists:
            print('{} already exists. Skipping step'.format(kwargs['output']))
            return
        return func(*args, **kwargs)
    return wrapper

def check_inputs(fasta, coverage):
    '''
    Check if all input files exist and have the right extension
    '''

    if fasta.suffix not in ['.fa', '.fasta', '.fna']:
        print('{} is not fasta formatted. Are the arguments in the right order?'.format(fasta))
        exit(42)

    if len(coverage) == 1:
        suffixes = coverage[0].suffixes
        ext = suffixes.pop()

        # if ext == '.gz':
        #     subprocess.check_output(['gunzip', coverage[0]])
        #     coverage[0] = coverage[0].replace('{}$'.format(ext, ''))
        #     ext = suffixes.pop()

        if ext == '.h5':
            coverage = coverage[0]

    else:
        for bam in coverage:
            if not bam.suffix == '.bam':
                print('{} is not bam formatted. Are the arguments in the right order?'.format(bam))
                exit(42)

def get_outputs(fasta, output, **kwargs) -> Dict[str, Path]:

    output.mkdir(exist_ok=True)

    output_files = {
        'filt_fasta': Path('{}/{}_filtered.fasta'.format(output, fasta.stem)),
        'filt_h5': Path('{}/coverage_filtered.h5'.format(output)),
        'singleton': Path('{}/singletons.txt'.format(output)),
        'pairs': {'train': Path('{}/pairs_train.npy'.format(output)),
                  'test': Path('{}/pairs_test.npy'.format(output))},
        'model': Path('{}/CoCoNet.pth'.format(output)),
        'nn_test': Path('{}/CoCoNet_test.csv'.format(output)),
        'repr': {'composition': Path('{}/representation_compo.h5'.format(output)),
                 'coverage': Path('{}/representation_cover.h5'.format(output))}
    }

    if 'hits_threshold' in kwargs:
        output_files.update({
            'adjacency_matrix': Path('{}/adjacency_matrix_nf{}.npy'.format(output, kwargs['n_frags'])),
            'refined_adjacency_matrix': Path('{}/adjacency_matrix_nf{}_refined.npy'.format(output, kwargs['n_frags'])),
            'assignments': Path('{}/leiden-{}-{}_nf{}.csv'.format(output, kwargs['hits_threshold'], kwargs['gamma1'], kwargs['n_frags'])),
            'refined_assignments': Path('{}/leiden-{}-{}-{}_nf{}.csv'.format(output, kwargs['hits_threshold'], kwargs['gamma1'], kwargs['gamma2'], kwargs['n_frags']))
        })

    return output_files

def get_input_shapes(kmer, fragment_length, rev_compl, h5, wsize, wstep):
    with h5py.File(h5, 'r') as handle:
        n_samples = handle.get(list(handle.keys())[0]).shape[0]

    input_shapes = {
        'composition': 4**kmer * (1-rev_compl) + 136 * rev_compl, # Fix the 136 with the good calculation
        'coverage': (ceil((fragment_length-wsize+1) / wstep), n_samples)
    }

    return input_shapes

def get_architecture(**kwargs):

    arch = {
        'composition': {'neurons': kwargs['compo_neurons']},
        'coverage': {'neurons': kwargs['cover_neurons'],
                     'n_filters': kwargs['cover_filters'],
                     'kernel_size': kwargs['cover_kernel'],
                     'conv_stride': kwargs['cover_stride']},
        'combined': {'neurons': kwargs['combined_neurons']}
    }
    return arch

@lru_cache(maxsize=None)
def kmer_rc_idx(k=4):
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
    kmer_encoding = sequence.translate(KMER_CODES)
    kmer_indices = [int(kmer_encoding[i:i+2*k], 2) for i in range(0, 2*(len(sequence)-k+1), 2)]

    return kmer_indices

def get_kmer_frequency(sequence, kmer=4, rc=False, index=False, norm=False):

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

def get_coverage(pairs, coverage_h5, window_size, window_step):
    h5data = h5py.File(coverage_h5)
    contigs = np.unique(pairs['sp'].flatten())

    try:
        coverage_data = {ctg: np.array(h5data.get(ctg)[:]) for ctg in contigs}
    except TypeError:
        print('Something went wrong')
        import ipdb;ipdb.set_trace()

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
        cov_sp = seen.get((species, start), None)

        if i%100 == 0:
            print("{:,}/{:,}".format(i, pairs.shape[0]), end='\r')

        if cov_sp is None:
            cov_sp = np.apply_along_axis(
                lambda x: avg_window(x, window_size, window_step),
                1,
                coverage_data[species][:, start:end]
            )
            seen[(species, start)] = cov_sp

        coverage_feature[sorted_idx[i]] = cov_sp

    return (coverage_feature[:n_pairs, :, :],
            coverage_feature[n_pairs:, :, :])

def avg_window(x, window_size, window_step):
    x_conv = np.convolve(x, np.ones(window_size)/window_size, mode="valid")
    return x_conv[::window_step]
