import pandas as pd
import numpy as np
from Bio.SeqIO.FastaIO import SimpleFastaParser

from util import get_kmer_frequency, avg_window

class CompositionGenerator(object):
    def __init__(self, fasta, pairs_file, batch_size=64, k=4):
        self.i = 0
        self.k = k
        self.genomes = { name: seq for name, seq in SimpleFastaParser(open(fasta)) }
        self.pairs = pd.read_csv(pairs_file).values
        self.batch_size = batch_size
        self.n_batches = int(len(self.pairs) / self.batch_size)

    def __iter__(self):
        return self

    def __len__(self):
        return self.n_batches

    def __next__(self):
        X1 = np.zeros([self.batch_size,4**self.k], dtype=np.uint32)
        X2 = np.zeros([self.batch_size,4**self.k], dtype=np.uint32)
        
        if self.i < self.n_batches:
            pairs_batch = self.pairs[self.i*self.batch_size:(self.i+1)*self.batch_size]
            
            for j,(spA,startA,endA,spB,startB,endB) in enumerate(pairs_batch):
                freqA = get_kmer_frequency(self.genomes[spA][startA:endA])
                freqB = get_kmer_frequency(self.genomes[spB][startB:endB])
                X1[j,:] = freqA
                X2[j,:] = freqB

            self.i += 1

            return X1,X2
        else:
            raise StopIteration()


class CoverageGenerator(object):
    def __init__(self, pairs_file, batch_size=64, avg_len=16):
        self.i = 0
        self.avg_len = avg_len
        self.pairs = pd.read_csv(pairs_file).values
        self.batch_size = batch_size
        self.n_batches = int(len(self.pairs) / self.batch_size)

    def __iter__(self):
        return self

    def __len__(self):
        return self.n_batches

    def __next__(self):
        X1 = np.zeros([self.batch_size,], dtype=np.uint32)
        X2 = np.zeros([self.batch_size,], dtype=np.uint32)
        
        if self.i < self.n_batches:
            pairs_batch = self.pairs[self.i*self.batch_size:(self.i+1)*self.batch_size]
            
            for j,(spA,startA,endA,spB,startB,endB) in enumerate(pairs_batch):
                X1[j,:] = 
                X2[j,:] = 

            self.i += 1

            return X1,X2
        else:
            raise StopIteration()
