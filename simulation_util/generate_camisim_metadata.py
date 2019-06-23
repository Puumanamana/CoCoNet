import os
import pandas as pd
from Bio import SeqIO

def gen_fasta():
    mapping = pd.read_csv("metadata.csv",index_col="accession")
    mapping.V_id = mapping.V_id.str.split("_").str.get(0)
    mapping = mapping.groupby(level=0)["V_id"].agg("first")

    genomes = { genome.id: genome
                for genome in SeqIO.parse("../data/database/viral.genomic_formated.fna","fasta") }

    for accession in mapping.index:
        SeqIO.write(genomes[accession],"camisim/contigs_fasta/{}.fasta".format(mapping[accession]),"fasta")

def gen_id_to_genome():
    mapping = pd.Series({ f.split(".")[0]: "camisim/contigs_fasta/{}".format(f) for f in os.listdir("camisim/contigs_fasta")})
    mapping.to_csv("camisim/id_to_genome.tsv",sep="\t")

def gen_metadata():

    meta_camisim = pd.read_csv("camisim/id_to_genome.tsv",sep="\t",index_col=0,header=None)
    
    metadata = pd.DataFrame(columns=["OTU","NCBI_ID","novelty_category"],
                            index=meta_camisim.index)
    metadata.index.name = "genome_ID"

    metadata["OTU"] = range(metadata.shape[0])
    metadata["NCBI_ID"] = 2
    metadata["novelty_category"] = 1

    metadata.to_csv("camisim/metadata_camisim.tsv",sep="\t")

if __name__ == '__main__':
    gen_fasta()
    gen_id_to_genome()
    gen_metadata()
