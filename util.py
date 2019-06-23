from os.path import splitext

import numpy as np

from Bio import SeqIO
from Bio.Seq import Seq

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

def avg_window(x,window_size):
    return np.convolve(x,np.ones(window_size)/window_size,
                       mode="valid")[::window_size]
