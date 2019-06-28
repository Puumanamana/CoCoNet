import pandas as pd
import numpy as np
from Bio import SeqIO
import torch

from util import get_kmer_frequency, get_coverage

class CompositionGenerator(object):
    def __init__(self, fasta, pairs_file, batch_size=64, k=4):
        self.i = 0
        self.k = k
        self.pairs = pd.read_csv(pairs_file,index_col=0,header=[0,1])
        self.batch_size = batch_size
        self.n_batches = max(1,int(len(self.pairs) / self.batch_size))

        self.set_genomes(fasta)

    def set_genomes(self,fasta):
        contigs = set(self.pairs.stack(level=0).sp)
        self.genomes = { contig.id: str(contig.seq)
                         for contig in SeqIO.parse(fasta,"fasta")
                         if contig.id in contigs }
        self.pairs = self.pairs.values

    def __iter__(self):
        return self

    def __len__(self):
        return self.n_batches

    def __next__(self):
        X1 = np.zeros([self.batch_size,4**self.k], dtype=np.float32)
        X2 = np.zeros([self.batch_size,4**self.k], dtype=np.float32)
        
        if self.i < self.n_batches:
            pairs_batch = self.pairs[self.i*self.batch_size:(self.i+1)*self.batch_size]
            
            for j,(spA,startA,endA,spB,startB,endB) in enumerate(pairs_batch):
                freqA = get_kmer_frequency(self.genomes[spA][startA:endA])
                freqB = get_kmer_frequency(self.genomes[spB][startB:endB])
                X1[j,:] = freqA
                X2[j,:] = freqB

            self.i += 1

            return torch.from_numpy(X1),torch.from_numpy(X2)
        else:
            raise StopIteration()


class CoverageGenerator(object):
    """
    Genearator for coverage data. 
    It loads the coverage every [load_batch] batches.
    """
    def __init__(self, h5_coverage, pairs_file,
                 batch_size=64, load_batch=1000,window_size=16):
        self.i = 0
        self.pairs = pd.read_csv(pairs_file,index_col=0,header=[0,1])
        self.h5_coverage = h5_coverage
        self.batch_size = batch_size
        self.n_batches = max(1,int(len(self.pairs) / self.batch_size))
        self.load_batch = load_batch
        self.window_size = window_size

    def __iter__(self):
        return self

    def __len__(self):
        return self.n_batches

    def load(self):
        print("Loading next coverage batch")
        pairs = self.pairs.iloc[ self.i*self.load_batch*self.batch_size
                                 : (self.i+1)*self.load_batch*self.batch_size, :]
        self.X1, self.X2 = get_coverage(pairs,self.h5_coverage,self.window_size)

    def __next__(self):
        
        if self.i < self.n_batches:
            if self.i % self.load_batch == 0:
                self.load()

            self.i += 1
        
            batch_indices = range((self.i-1)*self.batch_size,
                                  min((self.i)*self.batch_size,self.X1.shape[0]))

            return (
                torch.from_numpy(self.X1[batch_indices,:,:]),
                torch.from_numpy(self.X2[batch_indices,:,:])
            )
        else:
            raise StopIteration()
