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
    n_samples = len(coverage_bam)
    
    coverage_h5_tmp = h5py.File("/tmp/coverage.h5",'w')

    for i,bam in enumerate(coverage_bam):
        group = "sample_{}".format(i)
        print("Processing {}".format(group))

        depth_buffer = np.zeros(1)
        cur_ctg = ""
        
        coverage_h5_tmp.create_group(group)

        depth_file = bam_to_depth(bam)
        with open(depth_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='\t')
            
            for ctg,pos,d in csv_reader:
                if ctg != cur_ctg:
                    coverage_h5_tmp.create_dataset("{}/{}".format(group,ctg),depth_buffer)
                    depth_buffer = np.zeros(ctg_info[ctg],dtype=np.uint32)

                depth_buffer[pos-1] = d

            coverage_h5_tmp.create_dataset("{}/{}".format(group,ctg),depth_buffer)

        # Save eerything in a [N_samples,Genome_size] matrix
        coverage_h5 = h5py.File(output,'w')

        for ctg in ctg_info:
            try:
                coverage_samples = [ coverage_h5_tmp.get("sample_{}/{}".format(i,ctg))
                                     for i in range(len(coverage_bam)) ]
            except:
                import ipdb;ipdb.set_trace()
            coverage = np.hstack([cov for cov in coverage_samples])
            coverage_h5.create_dataset(ctg,coverage_h5)
                        
                
                
        # depth = pysam.depth(bam,'-d','200000')
        # depth = [ x.split("\t") for x in depth.split('\n')
        #           if len(x) > 0 ]
        # depth = [ list(g) for k,g in groupby(depth,key=lambda x:x[0]) ]
        
        # for j,d in enumerate(depth):
        #     ctg = d[0][0]
        #     tmp_file = "/tmp/{}.npy".format(ctg)

        #     if not os.path.exists(tmp_file):
        #         coverage = np.zeros([n_samples,ctg_info[ctg]]).astype(int)
        #     else:
        #         coverage = np.load(tmp_file)
            
        #     data = np.array(d)
        #     pos = data[:,1].astype(int)
        #     pos_cov = data[:,2].astype(int)

        #     coverage[i,pos-1] = pos_cov

        #     np.save(tmp_file,coverage)
            
        #     print("{:,}/{:,}".format(j,len(depth)),end='\r')

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

