from pathlib import Path
from Bio.SeqIO.FastaIO import SimpleFastaParser
import numpy as np

from coconet.core.feature import Feature
from coconet.tools import run_if_not_exists


class CompositionFeature(Feature):

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
    def filter_by_length(self, output=None, min_length=None):
        with open(str(output), 'w') as writer:
            for (ctg_id, seq) in self.get_iterator('fasta'):
                ctg_no_n = seq.upper().replace('N', '')

                if len(ctg_no_n) >= min_length:
                    writer.write(f'>{ctg_id}\n{ctg_no_n}\n')

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

    def get_valid_nucl_pos(self):
        '''
        Return position of ACGT only in fasta
        (useful to clean coverage data since N are discarded from the fasta)
        '''
        filt_ids = set(self.get_contigs('filt_fasta'))

        for (ctg_id, seq) in self.get_iterator('fasta'):
            if ctg_id in filt_ids:
                positions = np.fromiter(seq, (np.unicode, 1)) != 'N'
                yield (ctg_id, positions)

