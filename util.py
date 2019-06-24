from os.path import splitext
from time import time

import h5py
import numpy as np

from Bio import SeqIO
from Bio.Seq import Seq

def timer(func):
    def wrapper(*args,**kwargs):
        t0 = time()
        res = func(*args,**kwargs)
        duration = time()-t0
        
        print("{}: {} s".format(func.__name__, duration))
        return res
        
    return wrapper

def format_assembly(db_path,min_len=2048,output=None):

    if output is None:
        base, ext = splitext(db_path)
        output = "{}_formated{}".format(base,ext)

    formated_assembly = []
    for genome in SeqIO.parse(db_path,"fasta"):
        new_genome = genome
        new_genome.seq = Seq(str(genome.seq).replace('N',''))
        if len(new_genome.seq) > min_len:
            formated_assembly.append(new_genome)

    SeqIO.write(formated_assembly,output,"fasta")


kmer_codes = { ord('A'): '00', ord('C'): '01', ord('G'): '10', ord('T'): '11'}

def get_kmer_number(sequence,k=4):
    kmer_encoding = sequence.translate(kmer_codes)
    kmer_indices = [ int(kmer_encoding[i:i+2*k],2) for i in range(0,2*(len(sequence)-k+1),2) ]

    return kmer_indices

def get_kmer_frequency(sequence,k=4):
    kmer_indices = get_kmer_number(sequence,k=4)

    frequencies = np.bincount(kmer_indices,minlength=4**k)

    return frequencies

def get_coverage(pairs,coverage_h5,window_size,
                 columns=["sp","start","end"]):
    h5data = h5py.File(coverage_h5)
    contigs = np.unique(np.concatenate((pairs.sp1,pairs.sp2)))
    
    coverage_data = { ctg: h5data.get(ctg) for ctg in contigs }

    pairs_sorted = pairs.stack(level=0).sort_values(by=columns)
    order = np.argsort(pairs_sorted.idx)
    
    # Calculate coverage for this order
    frag_len,n_samples = list(coverage_data.values())[0].shape
    X = np.zeros([len(pairs_sorted),int(frag_len/window_size),n_samples])
    
    seen = {}

    for i,(sp,start,end) in enumerate(pairs_sorted[columns].values):
        cov_sp = cov.get((sp,start),None)

        if cov_sp is None:
            cov_sp = avg_window(coverage_data[sp][start:end,:], window_size)
            seen[(sp,start)] = cov_sp

        X[i] = cov_sp

    return (X[order[:pairs.shape[0]],:,:],
            X[order[pairs.shape[0]:],:,:])
    

def avg_window(x,window_size):
    return np.convolve(x,np.ones(window_size)/window_size,
                       mode="valid")[::window_size]
