import os
import csv
import subprocess

import h5py
import numpy as np
from Bio import SeqIO

def length_filter(fasta,output,min_length=2048):
    assembly_gt_minlen = [ contig for contig in SeqIO.parse(fasta,"fasta")
                           if len(contig.seq) >= min_length ]
    SeqIO.write(assembly_gt_minlen,output,"fasta")

def bam_to_depth(bam):
    output = "/tmp/{}".format(os.path.basename(bam).replace(".bam",".txt"))
    cmd = ["samtools", "depth", "-d", "20000",bam]

    with open(output, "w") as outfile:
        subprocess.call(cmd, stdout=outfile)

    return output

def bam_to_h5(fasta,coverage_bam,output):
    ctg_info = { seq.id: len(seq.seq)
                 for seq in SeqIO.parse(fasta,"fasta") }

    coverage_h5_tmp = h5py.File("/tmp/coverage.h5",'w')

    for i,bam in enumerate(sorted(coverage_bam)):
        group = "sample_{}".format(i)
        print("Processing {}".format(group))

        coverage_h5_tmp.create_group(group)

        depth_file  = bam_to_depth(bam)
        cur_ctg = ""

        with open(depth_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='\t')

            for i,(ctg,pos,d) in enumerate(csv_reader):
                if ctg != cur_ctg:
                    if i > 0:
                        coverage_h5_tmp.create_dataset("{}/{}".format(group,cur_ctg),data=depth_buffer)
                        
                    ctg_len = ctg_info.get(ctg,None)
                    cur_ctg = ctg

                    if ctg_len is None:
                        continue
                    
                    depth_buffer = np.zeros(ctg_info[ctg],dtype=np.uint32)

                depth_buffer[int(pos)-1] = int(d)

        coverage_h5_tmp.create_dataset("{}/{}".format(group,ctg),data=depth_buffer)

    # Save everything in a [N_samples,Genome_size] matrix
    coverage_h5 = h5py.File(output,'w')

    for ctg in ctg_info:
        coverage_samples = [ coverage_h5_tmp.get("sample_{}/{}".format(i,ctg)[:])
                             for i in range(len(coverage_bam)) ]
        coverage_samples = [ x if x is not None else np.zeros(ctg_info[ctg],dtype=np.uint32)
                             for x in coverage_samples ]

        coverage = np.vstack([cov for cov in coverage_samples])
        coverage_h5.create_dataset(ctg,data=coverage)
