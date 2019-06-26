import os
import numpy as np
import pandas as pd

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

from progressbar import progressbar

data_dir = os.path.expanduser('~/database')

def format_database(db_name="viral.genomic.fna",min_ctg_len=2048):
    mapping = { 'R': ['A','G'],
                'Y': ['C','T'],
                'S': ['G','C'],
                'W': ['A','T'],
                'K': ['G','T'],
                'M': ['A','C'],
                'B': ['C','G','T'],
                'D': ['A','G','T'],
                'H': ['A','C','T'],
                'V': ['A','C','G']
    }

    clean_sequences = []

    db_path = "{}/{}".format(data_dir,db_name)

    print("Cleaning {}".format(db_path))
    
    for i,seq in progressbar(enumerate(SeqIO.parse(db_path,"fasta"))):
        nucl_seq = str(seq.seq).replace('N','')

        clean_list = [ char if char in ["A","C","G","T"]
                       else np.random.choice(mapping[char])
                       for char in list(nucl_seq) ]
        if len(clean_list) < min_ctg_len:
            continue
        
        clean_seq = SeqRecord(Seq(''.join(clean_list)),
                              id=seq.id,
                              name=seq.name,
                              description=seq.description)
        clean_sequences.append(clean_seq)

    SeqIO.write(clean_seq,"{}/viral.genomic.ACGT.fasta".format(data_dir),"fasta")

def split_genome(idx, name, seq, min_ctg_len=2048):
    nb_frags = np.random.randint(1, int(len(seq)/min_ctg_len))

    contig_lengths = np.zeros(nb_frags)
    for i in range(nb_frags-1):
        # Random length but let enough length for the remaining contigs
        remaining_length = len(seq) - sum(contig_lengths)
        contig_lengths[i] = np.random.randint(min_ctg_len,
                                              remaining_length - (nb_frags-i-1)*min_ctg_len)

    ends = np.cumsum(contig_lengths).astype(np.uint32)
    # We extend the last contig until the end
    ends[-1] = len(seq)
    starts = np.concatenate((np.zeros(1), ends[0:-1])).astype(np.uint32)

    index = [ "V{}_{}".format(idx, k) for k in range(nb_frags) ]

    return pd.DataFrame({"start": starts, "end": ends, "length": ends-starts, "accession": name},
                        index=index)

def run(db_name="viral.genomic.ACGT.fasta"):
    db_path = "{}/{}".format(data_dir,db_name)
    
    genomes = [ seq for seq in SeqIO.parse(db_path,"fasta") ]
    cuts = [ split_genome(i, genome.id, str(genome.seq))
             for i,genome in enumerate(genomes) ]

    print("Extracting contigs")
    assembly = []
    for i,cut in enumerate(cuts):
        print("{:,}/{:,}".format(i,len(cuts)),end="\r")
        
        virus = str(genomes[i].seq)
        contigs = [ SeqRecord(
            Seq(virus[row.start:row.end]),
            id=V_id, name=V_id, description=row.accession
        ) for V_id, row in cut.iterrows() ]
        
        assembly += contigs

    SeqIO.write(assembly,"assembly.fasta","fasta")

    cuts = pd.concat(cuts)[["start","end","length","accession"]]
    cuts.index.name = "V_id"
    cuts.to_csv("metadata.csv")

if __name__ == '__main__':
    format_database(min_ctg_len=2048)
    run()
