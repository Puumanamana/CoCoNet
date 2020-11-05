'''
Functions to generate test data
'''

from math import ceil
from pathlib import Path
from tempfile import mkdtemp

import h5py
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from coconet.dl_util import initialize_model

TMP_DIR = mkdtemp()

FL = 20
STEP = 2
WSIZE = 3
WSTEP = 2

TEST_LEARN_PRMS = {'batch_size': 4, 'learning_rate': 1e-3,
                   'kmer': 4, 'rc': True,
                   'wsize': WSIZE, 'wstep': WSTEP, 'load_batch': 30}
TEST_CTG_LENGTHS = [60, 100, 80]

TEST_ARCHITECTURE = {
    'composition': {'neurons': [8, 4]},
    'coverage': {'neurons': [8, 4],
                 'n_filters': 2,
                 'kernel_size': 3,
                 'conv_stride': 2},
    'merge': {'neurons': 2}
}

TEST_SHAPES = {'composition': 136,
               'coverage': (ceil((FL - WSIZE+1) / WSTEP), 2)}

def generate_h5_file(*lengths, n_samples=2, filename=None,
                           baselines=None, empty_samples=None):
    '''
    - Generate coverage matrix
    - Saves it locally
    '''

    if baselines is None:
        baselines = [10] * len(lengths)

    filepath = '{}/{}'.format(TMP_DIR, filename)

    coverage = {}

    for i, (length, bl) in enumerate(zip(lengths, baselines)):
        coverage['V{}'.format(i)] = bl + np.random.normal(0, bl//5, [n_samples, length])

    if empty_samples is not None:
        for i, null_entries in enumerate(empty_samples):
            coverage['V{}'.format(i)][null_entries, :] = 0

    if filename is None:
        return coverage

    handle = h5py.File(filepath, 'w')
    for key, val in coverage.items():
        handle.create_dataset(key, data=val.astype(np.float32))
    handle.close()

    return Path(filepath)

def generate_fasta_file(*lengths, filename='sequences.fasta', save=True):
    '''
    - Generate fasta sequences
    - Saves it locally
    '''

    filepath = '{}/{}'.format(TMP_DIR, filename)

    contigs = []
    for i, length in enumerate(lengths):
        sequence = ''.join(np.random.choice(list('ACGT'), length).tolist())
        contigs.append(
            SeqRecord(id='V{}'.format(i), seq=Seq(sequence))
        )

    if save:
        SeqIO.write(contigs, filepath, 'fasta')
        return Path(filepath)

    return contigs

def generate_pair_file(filename='pairs.npy', save=True):
    '''
    - Generate pairs
    - Saves it locally or not
    '''

    filepath = '{}/{}'.format(TMP_DIR, filename)

    pairs = np.recarray([2, 2], dtype=[('sp', '<U10'), ('start', 'uint32'), ('end', 'uint32')])
    pairs['sp'] = [["V0", "V0"], ["V0", "V1"]]
    pairs['start'] = [[0, 10], [5, 0]]
    pairs['end'] = [[10, 20], [15, 10]]

    if save:
        np.save(filepath, pairs)
        return Path(filepath)

    return pairs

def generate_rd_model():

    model = initialize_model('CoCoNet', TEST_SHAPES, TEST_ARCHITECTURE)

    return model
