from multiprocessing.pool import Pool
from functools import partial

from tqdm import tqdm
import numpy as np
from Bio import SeqIO
import torch

from coconet.tools import get_kmer_frequency, get_coverage

class CompositionGenerator:
    def __init__(self, pairs_file, fasta=None,
                 batch_size=64, kmer=4, rc=False, norm=False, ncores=10):
        self.i = 0
        self.kmer = kmer
        self.pairs = np.load(pairs_file)
        self.batch_size = batch_size
        self.n_batches = max(1, int(len(self.pairs) / self.batch_size))
        self.rc = rc
        self.norm = norm

        self.set_genomes(fasta)
        self.pool = Pool(ncores)

    def set_genomes(self, fasta):
        contigs = np.unique(self.pairs['sp'].flatten())
        self.genomes = {contig.id: str(contig.seq)
                        for contig in SeqIO.parse(fasta, "fasta")
                        if contig.id in contigs}

    def __iter__(self):
        return self

    def __len__(self):
        return self.n_batches

    def __next__(self):

        if self.i < self.n_batches:
            pairs_batch = self.pairs[self.i*self.batch_size:(self.i+1)*self.batch_size]

            get_kmer_frequency_with_args = partial(
                get_kmer_frequency, kmer=self.kmer, rc=self.rc, norm=self.norm
            )
            fragments_a = [self.genomes[spA][startA:endA] for (spA, startA, endA), _ in pairs_batch]
            fragments_b = [self.genomes[spB][startB:endB] for _, (spB, startB, endB) in pairs_batch]

            x1 = self.pool.map(get_kmer_frequency_with_args, fragments_a)
            x2 = self.pool.map(get_kmer_frequency_with_args, fragments_b)

            self.i += 1

            return torch.FloatTensor(x1), torch.FloatTensor(x2)

        self.pool.close()
        raise StopIteration()

class CoverageGenerator:
    """
    Genearator for coverage data
    It loads the coverage every [load_batch] batches.
    """
    def __init__(self, pairs_file, coverage_h5=None,
                 batch_size=64, load_batch=1000, wsize=16, wstep=8):
        self.i = 0
        self.pairs = np.load(pairs_file)
        self.coverage_h5 = coverage_h5
        self.batch_size = batch_size
        self.n_batches = max(1, len(self.pairs) // self.batch_size)
        self.load_batch = load_batch
        self.wsize = wsize
        self.wstep = wstep

        self.pbar = None
        if self.n_batches > 1:
            self.pbar = tqdm(total=len(self.pairs), position=0, ncols=100,
                             bar_format="{percentage:3.0f}%|{bar:20} {postfix}")
            self.pbar.set_postfix_str('data loader')

    def __iter__(self):
        return self

    def __len__(self):
        return self.n_batches

    def load(self):
        '''
        Extract coverage for next pair batch
        '''

        pairs = self.pairs[self.i*self.batch_size : (self.i + self.load_batch)*self.batch_size]
        self.x1, self.x2 = get_coverage(pairs, self.coverage_h5, self.wsize, self.wstep, pbar=self.pbar)

    def __next__(self):
        if self.i < self.n_batches:
            if self.i % self.load_batch == 0:
                self.load()

            idx_inf = (self.i % self.load_batch) * self.batch_size
            idx_sup = idx_inf + self.batch_size

            self.i += 1

            return (
                torch.from_numpy(self.x1[idx_inf:idx_sup, :, :]),
                torch.from_numpy(self.x2[idx_inf:idx_sup, :, :])
            )

        raise StopIteration()
