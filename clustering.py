import numpy as np
import h5py
import torch

from Bio import SeqIO
from progressbar import progressbar

from util import get_kmer_number, avg_window

from time import time

def save_repr_all(model,n_frags,frag_len,kmer,window_size,
                  fasta=None,coverage_h5=None,outputs=None):

    cov_h5 = h5py.File(coverage_h5,'r')

    repr_h5 = { key: h5py.File(filename,'w') for key,filename in outputs.items() }    
    
    for contig in progressbar(SeqIO.parse(fasta,"fasta"), max_value=len(cov_h5)):
        step = int((len(contig)-frag_len) / n_frags)

        times = [time()]
        fragment_slices = np.array(
            [ np.arange(step*i, step*i+frag_len)
              for i in range(n_frags) ]
        )

        times.append(time())
        contig_kmers = np.array(get_kmer_number(str(contig.seq),k=kmer))[fragment_slices]
        times.append(time())
        
        x_composition = torch.from_numpy(np.array(
            [ np.bincount(indices,minlength=4**kmer)
            for indices in contig_kmers ]
        ).astype(np.float32)) # Shape = (n_frags, 4**k)
        times.append(time())
        
        coverage_genome = np.array(cov_h5.get(contig.id)[:]).astype(np.float32)[:,fragment_slices]
        coverage_genome = np.swapaxes(coverage_genome,1,0)
        times.append(time())
        
        x_coverage = torch.from_numpy(
            np.apply_along_axis(
                lambda x: avg_window(x,window_size), 2, coverage_genome
            ).astype(np.float32) )
        times.append(time())
        
        x_repr = model.compute_repr(x_composition,x_coverage)
        times.append(time())
        
        [ handle.create_dataset(contig.id, data=x_repr[key].detach().numpy(), dtype=np.float32)
          for key, handle in repr_h5.items() ]
        times.append(time())

        print(np.diff(times))
    [ handle.close() for handle in repr_h5.values() ]
        
def get_neighbors():
    pass

def cluster():
    pass
