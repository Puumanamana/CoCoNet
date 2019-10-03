from glob import glob
import os

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import h5py
import numpy as np


def split(l, fl=2048):
    if l > 2*fl and np.random.random() < 0.75:
        pos = np.random.randint(fl, l-fl)
        return [pos] + split(l-pos)

    return [l]

def apply_splits(seq, h5file, split_positions, min_coverage=-1):

    coverage = h5file.get(seq.id)[:]

    if (len(split_positions) == 1) or (coverage.mean(axis=1).max() < min_coverage):
        return {'sequences': [seq],
                'coverage': {seq.id: h5file.get(seq.id)[:]}}

    cumul_positions = np.cumsum([0]+split_positions)

    new_sequences = [
        SeqRecord(Seq(str(seq.seq)[cumul_positions[i]:cumul_positions[i+1]]),
                  id='{}|{}'.format(seq.id, i),
                  description='')
        for i in range(len(split_positions))
    ]

    new_coverage = {
        seq.id: coverage[:, cumul_positions[i]:cumul_positions[i+1]]
        for (i, seq) in enumerate(new_sequences)
    }

    output_data = {
        'sequences': new_sequences,
        'coverage': new_coverage
    }

    return output_data


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='.')
    parser.add_argument('--min-coverage', type=float, default=1)
    args = parser.parse_args()

    dataset = args.path

    output_dir = '{}_split_mc{:g}'.format(dataset, args.min_coverage)
    os.mkdir(output_dir)

    h5filename = glob('{}/coverage_contigs*.h5'.format(dataset))[0]
    handle = h5py.File(h5filename)
    assembly = {seq.id: seq for seq in
                SeqIO.parse('{}/assembly_gt2048.fasta'.format(dataset), 'fasta')}

    assembly_splits = {seq.id: split(len(seq.seq)) for seq in assembly.values()}
    results = [apply_splits(assembly[name], handle, splits, min_coverage=args.min_coverage)
               for name, splits in assembly_splits.items()]

    new_fasta = open('{}/assembly_gt2048.fasta'.format(output_dir), 'a')
    new_h5 = h5py.File('{}/coverage_contigs_gt2048.h5'.format(output_dir), 'w')

    for result in results:
        for name, cov in result['coverage'].items():
            new_h5.create_dataset(name, data=cov)
        SeqIO.write(result['sequences'], new_fasta, 'fasta')

    new_fasta.close()
    new_h5.close()
