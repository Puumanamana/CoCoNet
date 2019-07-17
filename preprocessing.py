import os
import h5py
import numpy as np
from Bio import SeqIO
import pysam
from itertools import groupby

def length_filter(fasta,output,min_length=2048):
    assembly_gt_minlen = [ contig for contig in SeqIO.parse(fasta,"fasta")
                           if len(contig.seq) >= min_length ]
    SeqIO.write(assembly_gt_minlen,output,"fasta")

def bam_to_h5(fasta,coverage_bam,output):
    ctg_info = { seq.id: len(seq.seq)
                 for seq in SeqIO.parse(fasta,"fasta") }
    n_samples = len(coverage_bam)

    for i,bam in enumerate(coverage_bam):
        depth = pysam.depth(bam,'-d','200000')
        depth = [ x.split("\t") for x in depth.split('\n')
                  if len(x) > 0 ]
        depth = [ list(g) for k,g in groupby(depth,key=lambda x:x[0]) ]
        
        for j,d in enumerate(depth):
            ctg = d[0][0]
            tmp_file = "/tmp/{}.npy".format(ctg)

            if not os.path.exists(tmp_file):
                coverage = np.zeros([n_samples,ctg_info[ctg]]).astype(int)
            else:
                coverage = np.load(tmp_file)
            
            data = np.array(d)
            pos = data[:,1].astype(int)
            pos_cov = data[:,2].astype(int)

            coverage[i,pos-1] = pos_cov

            np.save(tmp_file,coverage)
            
            print("{:,}/{:,}".format(j,len(depth)),end='\r')

    coverage_h5 = h5py.File(output)

    for ctg in ctg_info:
        tmp_file = "/tmp/{}.npy".format(ctg)

        if os.path.exists(tmp_file):
            coverage = np.load(tmp_file)
            os.remove(tmp_file)
        else:
            coverage = np.zeros([n_samples,ctg_info[ctg]]).astype(int)
            
        coverage_h5.create_dataset(ctg,data=coverage)
    coverage_h5.close()
    
# def bam_to_h5(fasta,coverage_bam,output):
#     ctg_info = [ (seq.id, len(seq.seq))
#                  for seq in SeqIO.parse(fasta,"fasta") ]
#     n_samples = len(coverage_bam)

#     coverage_h5 = h5py.File(output)

#     for ctg, length in progressbar(ctg_info):
#         coverage = np.zeros([n_samples,length])
        
#         for i,bam in enumerate(coverage_bam):
#             depth = get_depth(bam,ctg,length)

#             if depth.size > 0:
#                 positions = depth[:,1].astype(int)-1
#                 cov = depth[:,2].astype(int)
#                 coverage[i,positions] = cov

#         coverage_h5.create_dataset(ctg,data=coverage)

# def get_depth(bam_file,ctg,length):
    
#     target_region = "{}:1-{}".format(ctg,length)
#     depth = pysam.depth(bam_file, '-r', target_region, '-d', '200000')

#     coverage = np.array([ x.split('\t') for x in depth.split('\n')
#                           if len(x) > 0 ])

#     return coverage

