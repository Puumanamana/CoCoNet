import os
import pandas as pd
from Bio import SeqIO

import configparser

sim_dir = "../../../simulation"
fasta_dirname = "contigs_fasta"
sim_fasta_dir = "{}/{}".format(sim_dir,fasta_dirname)

def gen_fasta(database,out_dir=sim_fasta_dir):
    
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        
    [ SeqIO.write(genome,"{}/{}.fasta".format(out_dir,genome.id),"fasta")
      for genome in SeqIO.parse(database,"fasta") ]

def gen_id_to_genome(out_dir=sim_dir,fasta_dirname=fasta_dirname):
    mapping = pd.Series({ f.split(".")[0]: "{}/{}".format(fasta_dirname,f)
                          for f in os.listdir("{}/{}".format(out_dir,fasta_dirname))})
    mapping.to_csv("{}/id_to_genome.tsv".format(out_dir),sep="\t",header=False)

def gen_metadata(out_dir=sim_dir):

    meta_camisim = pd.read_csv("{}/id_to_genome.tsv".format(out_dir),sep="\t",index_col=0,header=None)
    
    metadata = pd.DataFrame(columns=["OTU","NCBI_ID","novelty_category"],
                            index=meta_camisim.index)
    metadata.index.name = "genome_ID"

    metadata["OTU"] = range(metadata.shape[0])
    metadata["NCBI_ID"] = 2
    metadata["novelty_category"] = 1

    metadata.to_csv("{}/metadata_camisim.tsv".format(out_dir),sep="\t")

def gen_config(out_dir=sim_dir):
    db = os.path.expanduser("~/databases/viral.genomic.ACGT.fasta")
    
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
        'readsim': 'CAMISIM-0.2.2/tools/art_illumina-2.3.6/art_illumina',
        'error_profiles': 'CAMISIM-0.2.2/tools/art_illumina-2.3.6/profiles',
        'samtools': '/home/cedric/.local/bin/samtools-1.9/samtools',
        'profile': 'mbarc',
        'size': 3,
        'type': 'art',
        'fragments_size_mean': 400,
        'fragment_size_standard_deviation': 10
    }
    config['CommunityDesign'] = {
        'ncbi_taxdump': 'CAMISIM-0.2.2/tools/ncbi-taxonomy_20190708.tar.gz',
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
    with open('{}/config.ini'.format(out_dir), 'w') as configfile:
        config.write(configfile)
    
if __name__ == '__main__':
    gen_config()
    gen_fasta("/home/cedric/databases/viral.genomic.ACGT.fasta")
    gen_id_to_genome()
    gen_metadata()
