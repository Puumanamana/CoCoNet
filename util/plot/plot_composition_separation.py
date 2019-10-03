import os
import sys
import pandas as pd
from Bio import SeqIO

from sklearn.metrics.pairwise import paired_distances
import seaborn as sns
import matplotlib.pyplot as plt

PARENT_DIR = os.path.join(sys.path[0], '../..')
sys.path.insert(0, PARENT_DIR)

from tool import get_kmer_frequency
from fragmentation import make_pairs
from generators import CompositionGenerator

def compute_distances(pair_file, fasta, kmer_list=[4], rc=False, norm=True, batch_size=None):
    kmer_generator = CompositionGenerator(
        pair_file, fasta, batch_size=batch_size,
        kmer_list=kmer_list, rc=rc, norm=norm)

    x_compo = map(lambda x: x.numpy(), next(kmer_generator))
    distances = paired_distances(*x_compo)

    return distances

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--db', type=str, default='/home/cedric/db/viral.genomic.ACGT.fasta')
    parser.add_argument('--fl', type=int, default=1024)
    args = parser.parse_args()

    pairs_out = '/tmp/cedric/pairs.npy'
    db_genomes = [seq for seq in SeqIO.parse(args.db, 'fasta')]

    distances_all = []
    for fl in [256, 512, 1024, 2048]:
        pairs = make_pairs(db_genomes, 64, fl,
                           n_examples=len(db_genomes)*5, output=pairs_out)

        truth = ['Within species' if sp1 == sp2 else 'Across species'
                 for sp1, sp2 in pairs['sp']]
        distances = compute_distances(pairs_out, args.db, batch_size=len(pairs))
        distances_all.append(
            pd.DataFrame({'dist': distances, 'truth': truth, 'fl': fl})
        )

    distances_all = pd.concat(distances_all)

    g = (
        sns.FacetGrid(data=distances_all, hue='truth', col='fl', col_wrap=2)
        .map(sns.kdeplot, 'dist')
        .add_legend(title='')
        .set(xlim=(0, 0.2), xticks=[])
        .set_titles('Fragment_length={col_name} bp')
        .set_axis_labels(x_var='Euclidean distance',
                         y_var='Count')
    )
    plt.show()


