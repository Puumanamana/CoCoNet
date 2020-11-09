"""
Generator for neural network training
"""

from multiprocessing.pool import Pool
from functools import partial

import numpy as np
from Bio.SeqIO.FastaIO import SimpleFastaParser
import torch

from coconet.tools import get_kmer_frequency, get_coverage

class CompositionGenerator:
    """
    kmer frequencies generator
    """

    def __init__(self, pairs_file, fasta=None,
                 batch_size=64, kmer=4, rc=False, threads=1):
        self.i = 0
        self.kmer = kmer
        self.pairs = np.load(pairs_file)
        self.n_batches = max(1, len(self.pairs) // max(batch_size, 1))
        self.batch_size = batch_size if batch_size > 0 else len(self.pairs)
        self.rc = rc

        if fasta is not None:
            self.set_genomes(fasta)
            self.pool = Pool(threads)

    def set_genomes(self, fasta):
        contigs = set(self.pairs['sp'].flatten())
        self.genomes = dict()

        with open(fasta, 'r') as handle:
            for (name, seq) in SimpleFastaParser(handle):
                ctg_id = name.split()[0]
                if ctg_id in contigs:
                    self.genomes[ctg_id] = seq

    def __iter__(self):
        return self

    def __len__(self):
        return self.n_batches

    def __next__(self):
        if self.i < self.n_batches:
            pairs_batch = self.pairs[self.i*self.batch_size:(self.i+1)*self.batch_size]

            get_kmer_frequency_with_args = partial(
                get_kmer_frequency, kmer=self.kmer, rc=self.rc
            )
            fragments_a = [self.genomes[spA][startA:endA] for (spA, startA, endA), _ in pairs_batch]
            fragments_b = [self.genomes[spB][startB:endB] for _, (spB, startB, endB) in pairs_batch]

            x1 = self.pool.map(get_kmer_frequency_with_args, fragments_a)
            x2 = self.pool.map(get_kmer_frequency_with_args, fragments_b)

            self.i += 1

            if self.i >= self.n_batches:
                self.pool.close()

            return torch.FloatTensor(x1), torch.FloatTensor(x2)

        raise StopIteration()

class CoverageGenerator:
    """
    Genearator for coverage feature
    It loads the coverage every [load_batch] batches.
    """
    def __init__(self, pairs_file, coverage_h5=None,
                 batch_size=64, load_batch=1000, wsize=16, wstep=8):
        self.i = 0
        self.pairs = np.load(pairs_file)
        self.coverage_h5 = coverage_h5
        self.n_batches = max(1, len(self.pairs) // max(batch_size, 1))
        self.batch_size = batch_size if batch_size > 0 else len(self.pairs)
        self.load_batch = load_batch
        self.wsize = wsize
        self.wstep = wstep

    def __iter__(self):
        return self

    def __len__(self):
        return self.n_batches

    def load(self):
        """
        Extract coverage for next pair batch
        """

        pairs = self.pairs[self.i*self.batch_size : (self.i + self.load_batch)*self.batch_size]

        self.x1, self.x2 = get_coverage(pairs, self.coverage_h5, self.wsize, self.wstep)

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
