import os
import pandas as pd
from Bio import SeqIO

import configparser

def gen_fasta(database):
    
    [ SeqIO.write(genome,"camisim/contigs_fasta/{}.fasta".format(genome.id),"fasta")
      for genome in SeqIO.parse(database,"fasta") ]

def gen_id_to_genome():
    mapping = pd.Series({ f.split(".")[0]: "contigs_fasta/{}".format(f) for f in os.listdir("camisim/contigs_fasta")})
    mapping.to_csv("camisim/id_to_genome.tsv",sep="\t",header=False)

def gen_metadata():

    meta_camisim = pd.read_csv("camisim/id_to_genome.tsv",sep="\t",index_col=0,header=None)
    
    metadata = pd.DataFrame(columns=["OTU","NCBI_ID","novelty_category"],
                            index=meta_camisim.index)
    metadata.index.name = "genome_ID"

    metadata["OTU"] = range(metadata.shape[0])
    metadata["NCBI_ID"] = 2
    metadata["novelty_category"] = 1

    metadata.to_csv("camisim/metadata_camisim.tsv",sep="\t")

def gen_config():
    db = os.path.expanduser("~/database/viral.genomic.ACGT.fasta")
    
    n_genomes = sum(1 for line in open(db) if line.startswith(">"))
    
    config = configparser.ConfigParser()
    config['Main'] = {
        'seed': 42,
        'phase': 0,
        'max_processors': 15,
        'dataset_id': 'vir_sim',
        'output_directory': 'sim',
        'temp_directory': '/tmp/',
        'gsa': False,
        'pooled_gsa': False,
        'anonymous': False,
        'compress': 0
    }
    config['ReadSimulator'] = {
        'readsim': '/home/cedric/.local/prog/CAMISIM/tools/art_illumina-2.3.6/art_illumina',
        'error_profiles': '/home/cedric/.local/prog/CAMISIM/tools/art_illumina-2.3.6/profiles',
        'samtools': '/home/cedric/.local/bin/samtools',
        'profile': 'mbarc',
        'size': 3,
        'type': 'art',
        'fragments_size_mean': 400,
        'fragment_size_standard_deviation': 10
    }
    config['CommunityDesign'] = {
        'ncbi_taxdump': '/home/cedric/.local/prog/CAMISIM/tools/ncbi-taxonomy_20180226.tar.gz',
        'number_of_samples': 5
    }
    config['community0'] = {
        'metadata': 'metadata_camisim.tsv',
        'id_to_genome_file': 'id_to_genome.tsv',
        'genomes_total': n_genomes,
        'genomes_real': n_genomes,
        'max_strains_per_otu': 1,
        'ratio': 1,
        'mode': 'differential',
        'log_mu': 1,
        'log_sigma': 2
    }
    with open('camisim/config.ini', 'w') as configfile:
        config.write(configfile)
    
if __name__ == '__main__':
    gen_config()
    gen_fasta("/home/cedric/database/viral.genomic.ACGT.fasta")
    gen_id_to_genome()
    gen_metadata()
