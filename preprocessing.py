import os
import shutil
from tempfile import mkdtemp
import csv
import subprocess

import h5py
import numpy as np

from Bio import SeqIO
from Bio.Seq import Seq

from progressbar import progressbar

def format_assembly(fasta,output=None,min_length=2048):

    if output is None:
        base, ext = os.path.splitext(fasta)
        output = "{}_gt{}{}".format(base,min_length,ext)

    formated_assembly = [ genome for genome in SeqIO.parse(fasta,"fasta")
                          if len(str(genome.seq).replace('N','')) >= min_length]

    SeqIO.write(formated_assembly,output,"fasta")    

def bam_to_h5(bam,temp_dir,ctg_info):
    """
    - Run samtools depth on bam file and save it in temp_dir
    - Read the output and save the result in a h5 file with keys as contigs
    """
    
    file_rad = os.path.splitext(os.path.basename(bam))[0]
    txt_output = "{}/{}.txt".format(temp_dir,file_rad)
    h5_output = "{}/{}.h5".format(temp_dir,file_rad)
    
    cmd = ["samtools", "depth", "-d", "20000",bam]

    with open(txt_output, "w") as outfile:
        subprocess.call(cmd, stdout=outfile)

    n_entries = sum(1 for _ in open(txt_output))
    h5_handle = h5py.File(h5_output,'w') 
    
    # Save the coverage of each contig in a separate file
    current_ctg = None
    depth_buffer = None
    
    with open(txt_output,'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')

        for i,(ctg,pos,d) in progressbar(enumerate(csv_reader), max_value=n_entries):
            # 1st case: contig is not in the assembly (filtered out in previous step)
            ctg_len = ctg_info.get(ctg,None)

            if ctg_len is None:
                continue
                
            # 2nd case: it's a new contig in the depth file
            if ctg != current_ctg:
                # If it's not the first one, we save it
                if current_ctg is not None :
                    h5_handle.create_dataset("{}".format(current_ctg),
                                             data=depth_buffer)
                # Update current contigs
                current_ctg = ctg
                depth_buffer = np.zeros(ctg_len,dtype=np.uint32)

            # Fill the contig with depth info
            depth_buffer[int(pos)-1] = int(d)
        
    return h5_output

def bam_list_to_h5(fasta,coverage_bam,output,min_samples=2):
    """
    - Extract the coverage of the sequences in fasta from the bam files
    - Remove N nucleotides from the FASTA and remove the corresponding entries from coverage
    - Filter out the contigs with less than [min_samples] samples with a mean coverage less than 1
    """
    
    temp_dir = mkdtemp()
    ctg_info = { seq.id: len(seq.seq)
                 for seq in SeqIO.parse(fasta,"fasta") }

    depth_h5_files = [ bam_to_h5(bam,temp_dir,ctg_info) for bam in sorted(coverage_bam) ]

    # Collect everything in a [N_samples,genome_size] matrix
    coverage_h5 = h5py.File(output,'w')
    
    ctg_seq = { seq.id: seq
                for seq in SeqIO.parse(fasta,"fasta") }
    assembly_noN = []

    h5_handles = [ h5py.File(f,'r') for f in depth_h5_files ]

    for ctg in ctg_info:
        ctg_coverage = [ h.get(ctg) for h in h5_handles ]
        ctg_coverage = np.vstack([ x[:] if x is not None
                                   else np.zeros(ctg_info[ctg],dtype=np.uint32)
                                   for x in ctg_coverage ])

        # Take care of the N problem
        loc_ACGT = np.array([i for i,letter in enumerate(ctg_seq[ctg]) if letter != 'N'])

        ctg_coverage = ctg_coverage[:,loc_ACGT]

        # Filter out contig with coverage on only 1 sample
        mean_coverage = ctg_coverage.sum(axis=1)
        if sum(mean_coverage >= ctg_info[ctg]) < min_samples:
            continue

        coverage_h5.create_dataset(ctg,data=ctg_coverage)

        # Process the sequence
        seq_noN = ctg_seq[ctg]
        seq_noN.seq = Seq(str(seq_noN.seq).replace('N','').upper())
        
        assembly_noN.append(seq_noN)

    SeqIO.write(assembly_noN,fasta,"fasta")

    # Remove temp directory
    shutil.rmtree(temp_dir)
