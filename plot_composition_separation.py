import tempfile
import argparse
import os
import sys
import shutil
from pathlib import Path

import pandas as pd
from Bio import SeqIO

from sklearn.metrics.pairwise import paired_distances
import seaborn as sns
import matplotlib.pyplot as plt

PARENT_DIR = os.path.join(sys.path[0], '../..')
sys.path.insert(0, PARENT_DIR)

from fragmentation import make_pairs
from generators import CompositionGenerator

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', type=str, default='/home/cedric/databases/viral.genomic.ACGT.fasta')
    parser.add_argument('--fl', type=int, default=1024)
    parser.add_argument('--k', type=int, default=4)
    args = parser.parse_args()

    return args

def compute_distances(pair_file, fasta, kmer=4, rc=True, norm=True, batch_size=None):
    kmer_generator = CompositionGenerator(
        pair_file, fasta, batch_size=batch_size,
        kmer=kmer, rc=rc, norm=norm)

    x_compo = map(lambda x: x.numpy(), next(kmer_generator))
    distances = paired_distances(*x_compo)

    return distances

def run(fl, k, db, db_path):
    print('Processing fragment length = {}'.format(fl))
    pairs_out = tempfile.mkdtemp() + '/pairs.npy'
    
    pairs = make_pairs(db, 64, fl, n_examples=len(db)*5, output=pairs_out)
    truth = ['Within species' if sp1 == sp2 else 'Across species' for sp1, sp2 in pairs['sp']]
    distances = pd.DataFrame({
        'dist': compute_distances(pairs_out, db_path, kmer=k, batch_size=len(pairs)),
        'truth': truth,
        'fl': fl})

    shutil.rmtree(Path(pairs_out).parent)
    return distances


def main():
    args = parse_args()
    
    db_genomes = [seq for seq in SeqIO.parse(args.db, 'fasta')
                  if len(seq.seq) >= 2048]
    print('{} genomes'.format(len(db_genomes)))

    distances_all = []
    fls = [256, 512, 1024, 2048]

    results = [run(fl, args.k, db_genomes, args.db) for fl in fls]
    distances_all = pd.concat(results)
    
    sns.violinplot(x="fl", y="dist", hue="truth", data=distances_all, gridsize=500,
                   split=True, inner=None)
    sns.despine(left=True)

    plt.ylim([0, 0.2])
    plt.xlabel('Fragment length (bp)', fontsize=14)
    plt.ylabel('L2 Distance', fontsize=14)
    plt.legend(title='')

    plt.savefig('figures/Figure 1-Composition_separation_with_fl.pdf', transparent=True)
    plt.show()

if __name__ == '__main__':
    main()

