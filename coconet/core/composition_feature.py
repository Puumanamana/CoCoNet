from pathlib import Path
from Bio.SeqIO.FastaIO import SimpleFastaParser
import numpy as np

from coconet.core.feature import Feature
from coconet.tools import run_if_not_exists


class CompositionFeature(Feature):

    def __init__(self, **kwargs):
        Feature.__init__(self, **kwargs)
        self.ftype = 'composition'

    def get_iterator(self, key='fasta'):
        iterator = SimpleFastaParser(open(str(self.path[key]), 'r'))
        iterator = map(lambda x: [x[0].split()[0], x[1]], iterator)

        return iterator

    def get_contigs(self, key=None):
        if key is None:
            if 'latent' in self.path:
                key = 'latent'
            elif 'fasta' in self.path:
                key = 'fasta'
            else:
                raise ValueError("No contig information found")

        if 'fasta' in key:
            contigs =  [title.split()[0] for (title, _) in self.get_iterator(key=key)]
        elif key == 'latent':
            handle = self.get_handle(key)
            contigs = list(handle.keys())
            handle.close()
        return np.array(contigs)

    def write_bed(self, output=None):
        filt_ids = set(self.get_contigs('filt_fasta'))

        with open(str(output), 'w') as writer:
            for (ctg_id, seq) in self.get_iterator('fasta'):
                if ctg_id in filt_ids:
                       writer.write('\t'.join([ctg_id, '1', str(1+len(seq))]) + '\n')

    @run_if_not_exists()
    def filter_by_length(self, output=None, min_length=None):
        with open(str(output), 'w') as writer:
            for (ctg_id, seq) in self.get_iterator('fasta'):
                ctg_no_n = seq.upper().replace('N', '')

                if len(ctg_no_n) >= min_length:
                    writer.write(f'>{ctg_id}\n{ctg_no_n}\n')

        self.path['filt_fasta'] = Path(output)

    def filter_by_ids(self, output=None, ids_file=None):
        # Cannot stream if output is the same as the input
        filtered_fasta = []
        ids = {x.strip().split()[0] for x in open(ids_file)}

        for (ctg_id, seq) in self.get_iterator('filt_fasta'):
            if ctg_id not in ids:
                filtered_fasta.append((ctg_id, seq))

        with open(str(output), 'w') as writer:
            for (ctg_id, seq) in filtered_fasta:
                writer.write(f'>{ctg_id}\n{seq}\n')

    def summarize_filtering(self, singletons=None):
        n_before = sum(1 for _ in self.get_iterator('fasta'))
        n_after = sum(1 for _ in self.get_iterator('filt_fasta'))
        n_singletons = -1

        if singletons is not None and Path(singletons).is_file():
            n_singletons = sum(1 for _ in open(singletons)) - 1

        return f'before: {n_before:,}, after: {n_after:,} (#singletons={n_singletons:,})'

    def get_valid_nucl_pos(self):
        '''
        Return position of ACGT only in fasta
        (useful to clean coverage data since N are discarded from the fasta)
        '''
        filt_ids = set(self.get_contigs('filt_fasta'))

        for (ctg_id, seq) in self.get_iterator('fasta'):
            if ctg_id in filt_ids:
                positions = np.fromiter(seq, (np.unicode,1)) != 'N'
                yield (ctg_id, positions)
