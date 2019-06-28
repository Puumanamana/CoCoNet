import os
import numpy as np
import pandas as pd

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

from progressbar import progressbar

data_dir = os.path.expanduser('~/database')

def format_database(min_virus_len, db_name="viral.genomic.fna"):
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
    
    v_i = 0
    for i,seq in progressbar(enumerate(SeqIO.parse(db_path,"fasta"))):
        nucl_seq = str(seq.seq).replace('N','')

        clean_list = [ char if char in ["A","C","G","T"]
                       else np.random.choice(mapping[char])
                       for char in list(nucl_seq) ]
        if len(clean_list) < min_virus_len:
            continue

        virus_id = "V{}".format(v_i)
        clean_seq = SeqRecord(Seq(''.join(clean_list)),
                              id=virus_id,
                              name=virus_id,
                              description=seq.description)
        v_i += 1
        clean_sequences.append(clean_seq)
        
    SeqIO.write(clean_sequences,"{}/viral.genomic.ACGT.fasta".format(data_dir),"fasta")

def split_genome(genome, min_ctg_len):
    nb_frags = np.random.randint(1, 1+int(len(genome.seq)/min_ctg_len))

    contig_lengths = np.zeros(nb_frags, dtype=np.uint32)

    
    for i in range(nb_frags):
        # Random length but let enough length for the remaining contigs
        remaining_length = len(genome.seq) - sum(contig_lengths)
        contig_lengths[i] = np.random.randint(min_ctg_len,
                                              remaining_length - (nb_frags-i-1)*min_ctg_len + 1)
        if i == nb_frags-1:
            contig_lengths[i] = remaining_length

    ends = np.cumsum(contig_lengths).astype(np.uint32)
    
    starts = np.concatenate((np.zeros(1), ends[0:-1])).astype(np.uint32)

    index = [ "{}_{}".format(genome.id, k) for k in range(nb_frags) ]

    return pd.DataFrame({"start": starts, "end": ends, "length": ends-starts,
                         "accession": genome.description.split(' ')[1]},
                        index=index)

def run(min_ctg_len, db_name="viral.genomic.ACGT.fasta"):
    db_path = "{}/{}".format(data_dir,db_name)
    
    genomes = [ seq for seq in SeqIO.parse(db_path,"fasta") ]
    cuts = [ split_genome(genome,min_ctg_len) for genome in SeqIO.parse(db_path,"fasta") ]

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
    min_virus_len = 3000
    min_ctg_len = 2000
    # format_database(min_virus_len)
    run(min_ctg_len)
