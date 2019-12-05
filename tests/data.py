'''
Functions to generate test data
'''

from pathlib import Path

import h5py
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

LOCAL_DIR = Path(__file__).resolve().parent

def generate_coverage_file(n_samples=2, filename='coverage.h5'):
    '''
    - Generate coverage matrix
    - Saves it locally
    '''

    filepath = '{}/{}'.format(LOCAL_DIR, filename)

    cov = {'V0': np.tile(np.arange(50), (n_samples, 1)),
           'V1': np.tile(10, (n_samples, 50))}

    with h5py.File(filepath, 'w') as handle:
        for key, val in cov.items():
            handle.create_dataset(key, data=val)

    return Path(filepath)

def generate_fasta_file(*lengths, filename='sequences.fasta', save=True):
    '''
    - Generate fasta sequences
    - Saves it locally
    '''

    filepath = '{}/{}'.format(LOCAL_DIR, filename)

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

    filepath = '{}/{}'.format(LOCAL_DIR, filename)

    pairs = np.recarray([2, 2], dtype=[('sp', '<U10'), ('start', 'uint32'), ('end', 'uint32')])
    pairs['sp'] = [["V0", "V0"], ["V0", "V1"]]
    pairs['start'] = [[0, 10], [5, 0]]
    pairs['end'] = [[10, 20], [15, 10]]

    if save:
        np.save(filepath, pairs)
        return Path(filepath)

    return pairs
