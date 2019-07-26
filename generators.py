import numpy as np
from Bio import SeqIO
import torch

from util import get_kmer_frequency, get_coverage

class CompositionGenerator(object):
    def __init__(self, fasta, pairs_file, batch_size=64, kmer_list=4, rc=False):
        self.i = 0
        self.kmer_list = kmer_list
        self.pairs = np.load(pairs_file)
        self.batch_size = batch_size
        self.n_batches = max(1,int(len(self.pairs) / self.batch_size))
        self.rc = rc

        self.set_genomes(fasta)

    def set_genomes(self,fasta):
        contigs = np.unique(self.pairs['sp'].flatten())
        
        self.genomes = { contig.id: str(contig.seq)
                         for contig in SeqIO.parse(fasta,"fasta")
                         if contig.id in contigs }

    def __iter__(self):
        return self

    def __len__(self):
        return self.n_batches

    def __next__(self):
        feature_sizes = [0]+[int(4**k / (1+self.rc)) for k in self.kmer_list]
        # limits = np.cumsum(feature_sizes)
        # feature_size = self.pairs[0,2] - self.pairs[0,1] - self.k + 1
        
        X1 = np.zeros([self.batch_size,sum(feature_sizes)], dtype=np.float32)
        X2 = np.zeros([self.batch_size,sum(feature_sizes)], dtype=np.float32)
        
        if self.i < self.n_batches:
            pairs_batch = self.pairs[self.i*self.batch_size:(self.i+1)*self.batch_size]
            
            for j,((spA,startA,endA),(spB,startB,endB)) in enumerate(pairs_batch):
                freqA = get_kmer_frequency(self.genomes[spA][startA:endA],
                                           kmer_list=self.kmer_list,rc=self.rc)
                freqB = get_kmer_frequency(self.genomes[spB][startB:endB],
                                           kmer_list=self.kmer_list,rc=self.rc)
                
                # for k in range(2,self.k+1):
                #     # freqA = get_kmer_number(self.genomes[spA][startA:endA],k=k)
                #     # freqB = get_kmer_number(self.genomes[spB][startB:endB],k=k)
                #     freqA = get_kmer_frequency(self.genomes[spA][startA:endA],k=k,rc=self.rc)
                #     freqB = get_kmer_frequency(self.genomes[spB][startB:endB],k=k,rc=self.rc)

                X1[j,:] = freqA
                X2[j,:] = freqB
                # X1[j,limits[k-2]:limits[k-1]] = freqA
                # X2[j,limits[k-2]:limits[k-1]] = freqB

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
        self.pairs = np.load(pairs_file)
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
        print("\nLoading next coverage batch")
        
        pairs = self.pairs[ self.i*self.batch_size : (self.i + self.load_batch)*self.batch_size ]
        self.X1, self.X2 = get_coverage(pairs,self.h5_coverage,self.window_size)

    def __next__(self):
        
        if self.i < self.n_batches:
            if self.i % self.load_batch == 0:
                self.load()

            idx_inf = (self.i % self.load_batch) * self.batch_size
            idx_sup = idx_inf + self.batch_size
            
            self.i += 1
            
            return (
                torch.from_numpy(self.X1[idx_inf:idx_sup,:,:]),
                torch.from_numpy(self.X2[idx_inf:idx_sup,:,:])
            )
        else:
            raise StopIteration()
