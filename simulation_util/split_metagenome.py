from glob import glob
import os

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import h5py
import numpy as np


def split(L, fl=2048):
    if L>2*fl and np.random.random() < 0.75:
        pos = np.random.randint(fl, L-fl)
        return [pos] + split(L-pos)
    
    return [L]

def apply_splits(seq, h5file, split_positions):
    if len(split_positions) == 1:
        return {'sequences': [seq],
                'coverage': {seq.id: h5file.get(seq.id)[:]}}
    
    cumul_positions = np.cumsum([0]+split_positions)
    
    new_sequences = [
        SeqRecord(Seq(str(seq.seq)[cumul_positions[i]:cumul_positions[i+1]]),
                  id="{}|{}".format(seq.id, i),
                  description="")
        for i in range(len(split_positions))
    ]

    coverage = h5file.get(seq.id)[:]
    new_coverage =  {
        seq.id: coverage[:, cumul_positions[i]:cumul_positions[i+1]]
        for (i, seq) in enumerate(new_sequences)
    }

    output = {
        'sequences': new_sequences,
        'coverage': new_coverage
    }

    return output


if __name__ == '__main__':
    
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='.')
    args = parser.parse_args()

    dataset = args.path

    os.mkdir("{}_split".format(dataset))

    h5filename = glob('{}/coverage_contigs*.h5'.format(dataset))[0]
    handle = h5py.File(h5filename)
    assembly = {seq.id: seq for seq in SeqIO.parse('{}/assembly_gt2048.fasta'.format(dataset), 'fasta')}

    assembly_splits = {seq.id: split(len(seq.seq)) for seq in assembly.values()}
    results = [apply_splits(assembly[name], handle, splits)
               for name, splits in assembly_splits.items()]

    new_fasta = open("{}_split/assembly_gt2048.fasta".format(dataset), "a")
    new_h5 = h5py.File('{}_split/coverage_contigs_gt2048.h5'.format(dataset), 'w')

    for result in results:
        [new_h5.create_dataset(name, data=cov) for name, cov in result['coverage'].items()]
        SeqIO.write(result['sequences'], new_fasta, 'fasta')

    new_fasta.close()
    new_h5.close()

     
