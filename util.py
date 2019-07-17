from os.path import splitext
from time import time

import h5py
import numpy as np

from Bio import SeqIO
from Bio.Seq import Seq

from progressbar import progressbar

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

def get_kmer_frequency(sequence,k=4,rc=False,index=False):
    if not index:
        kmer_indices = get_kmer_number(sequence,k=4)
    else:
        kmer_indices = sequence

    frequencies = np.bincount(kmer_indices,minlength=4**k)

    if rc:
        frequencies += frequencies[::-1]
        frequencies = frequencies[:int(4**k/2)]

    return frequencies

def get_coverage(pairs,coverage_h5,window_size,
                 columns=["sp","start","end"]):
    h5data = h5py.File(coverage_h5)
    contigs = np.unique(np.concatenate((pairs.A.sp,
                                        pairs.B.sp)))

    coverage_data = { ctg: np.array(h5data.get(ctg)[:]) for ctg in contigs }

    pairs_sorted = pairs.stack(level=0).swaplevel().sort_index()
    pairs_sorted.index = np.arange(len(pairs_sorted))
    pairs_sorted.sort_values(by=columns,inplace=True)
    
    order = np.argsort(pairs_sorted.index)
    
    # Calculate coverage for this order
    n_samples = np.array(list(coverage_data.values())[0]).shape[0]
    frag_len = pairs_sorted.end.iloc[0] - pairs_sorted.start.iloc[0]
    
    X = np.zeros([len(pairs_sorted),n_samples,int(frag_len/window_size)],
                 dtype=np.float32)
    seen = {}

    for i,(sp,start,end) in progressbar(enumerate(pairs_sorted[columns].values),
                                        max_value=pairs_sorted.shape[0]):
        cov_sp = seen.get((sp,start),None)

        if cov_sp is None:
            cov_sp = np.apply_along_axis(
                lambda x: avg_window(x,window_size), 1, coverage_data[sp][:,start:end]
            )
            seen[(sp,start)] = cov_sp

        X[i] = cov_sp

    return (X[order[:pairs.shape[0]],:,:],
            X[order[pairs.shape[0]:],:,:])
    

def avg_window(x,window_size=4):
    return np.convolve(x,np.ones(window_size)/window_size,
                       mode="valid")[::window_size]
