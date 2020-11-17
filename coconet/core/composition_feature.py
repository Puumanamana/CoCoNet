"""
Composition feature manipulation
"""

from pathlib import Path
from Bio.SeqIO.FastaIO import SimpleFastaParser
import skbio
import numpy as np
import h5py

from coconet.core.feature import Feature
from coconet.tools import run_if_not_exists


class CompositionFeature(Feature):
    """
    Composition object routines
    """

    def __init__(self, **kwargs):
        Feature.__init__(self, **kwargs)
        self.name = 'composition'

    def count(self, key='fasta'):
        new_count = sum(1 for line in open(self.path[key]) if line.startswith('>'))
        return new_count

    def get_iterator(self, key='fasta'):
        iterator = SimpleFastaParser(open(str(self.path[key]), 'r'))
        iterator = map(lambda x: [x[0].split()[0], x[1]], iterator)

        return iterator

    def get_contigs(self, key=None):
        if key is None:
            if 'latent' in self.path:
                key = 'latent'
            elif 'filt_fasta' in self.path:
                key = 'filt_fasta'
            elif 'fasta' in self.path:
                key = 'fasta'
            else:
                raise ValueError("No contig information found")

        if 'fasta' in key:
            contigs =  [title.split()[0] for (title, _) in self.get_iterator(key=key)]
        elif key == 'latent':
            with h5py.File(self.path[key], 'r') as handle:
                contigs = list(handle.keys())
        return np.array(contigs)

    @run_if_not_exists()
    def filter_by_length(self, output=None, summary_output=None, min_length=None):
        summary_handle = open(summary_output, 'w')
        summary_handle.write('\t'.join(['contig', 'reason', 'value'])+'\n')

        with open(str(output), 'w') as writer:
            for (ctg_id, seq) in self.get_iterator('fasta'):
                ctg_no_n = seq.upper().replace('N', '')

                if len(ctg_no_n) >= min_length:
                    writer.write(f'>{ctg_id}\n{ctg_no_n}\n')
                else:
                    entry = '\t'.join([ctg_id, 'length', str(len(ctg_no_n))])
                    summary_handle.write(f'{entry}\n')

        summary_handle.close()

        self.path['filt_fasta'] = Path(output)

    def filter_by_ids(self, output=None, ids=None, ids_file=None):
        # Cannot stream if output is the same as the input
        filtered_fasta = []

        if ids_file is not None:
            ids = {x.strip().split()[0] for x in open(ids_file)}

        for (ctg_id, seq) in self.get_iterator('filt_fasta'):
            if ctg_id not in ids:
                filtered_fasta.append((ctg_id, seq))

        if output is None:
            output = self.path['filt_fasta']

        with open(str(output), 'w') as writer:
            for (ctg_id, seq) in filtered_fasta:
                writer.write(f'>{ctg_id}\n{seq}\n')

    @run_if_not_exists()
    def flag_dtr(self, output=None, key='filt_fasta', min_size=10, max_size=300, min_id=.95):
        handle = open(output, 'w')

        count = 0
        for ctg in skbio.io.read(str(self.path[key]), format='fasta', constructor=skbio.DNA):
            (aln, _, pos) = skbio.alignment.local_pairwise_align_ssw(
                ctg[:max_size], ctg[-max_size:], match_score=1, mismatch_score=-5
            )

            matches = sum(x==y for (x, y) in aln.iter_positions())

            if (aln.shape.position > min_size and
                matches / aln.shape.position >= min_id and
                pos[0][0] == 0 and pos[1][1] == max_size-1):
                # DTR found
                entry = [ctg.metadata["id"],
                         '-'.join(map(str, pos[0])),
                         '-'.join(map(str, pos[1]))]
                handle.write('\t'.join(entry) + '\n')
            else:
                count += 1

        handle.close()

        return count


    def get_valid_nucl_pos(self):
        """
        Return position of ACGT only in fasta
        (useful to clean coverage data since N are discarded from the fasta)
        """
        filt_ids = set(self.get_contigs('filt_fasta'))

        for (ctg_id, seq) in self.get_iterator('fasta'):
            if ctg_id in filt_ids:
                positions = np.fromiter(seq, (np.unicode, 1)) != 'N'
                yield (ctg_id, positions)
