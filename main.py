import os
from glob import glob

from Bio import SeqIO
import numpy as np
import h5py
import torch

from config import min_contig_length
from config import io_path
from config import frag_len,step,kmer_list,model_type
from config import n_examples
from config import nn_arch, train_args
from config import cluster_args

from preprocessing import format_assembly, bam_list_to_h5
from npy_fragmentation import make_pairs
from nn_training import initialize_model,train
from clustering import save_repr_all, cluster

def run():

    assembly_file = [ f for f in glob("{}/*.f*a".format(io_path["in"]))
                      if 'gt' not in f ][0]
    _, ext = os.path.splitext(assembly_file)
    
    
    raw_inputs = { "fasta": assembly_file,
                   "coverage_bam": glob("{}/*.bam".format(io_path["in"])) }
    
    filtered_inputs = { "fasta": assembly_file.replace(ext,"_gt{}{}".format(min_contig_length,ext)),
                        "coverage_h5": "{}/coverage_contigs.h5".format(io_path["in"]) }
    
    #######################
    #### Preprocessing ####
    #######################

    if not os.path.exists(filtered_inputs["coverage_h5"]):
        format_assembly(raw_inputs['fasta'],filtered_inputs['fasta'],min_length=min_contig_length)
        bam_list_to_h5(filtered_inputs['fasta'],raw_inputs['coverage_bam'],
                       output=filtered_inputs["coverage_h5"])  
        
    h5_cov = h5py.File(filtered_inputs['coverage_h5'],'r')
    n_samples = h5_cov.get(list(h5_cov.keys())[0]).shape[0]

    #######################
    #### Fragmentation ####
    #######################        

    pairs = {
        "test": "{}/pairs_test.npy".format(io_path["out"]),        
        "train": "{}/pairs_train.npy".format(io_path["out"])
    }

    if not os.path.exists(pairs["train"]):
        assembly = [ contig for contig in SeqIO.parse(filtered_inputs["fasta"],"fasta") ]

        assembly_idx = { 'test': np.random.choice(len(assembly),int(0.1*len(assembly))) }
        assembly_idx['train'] = np.setdiff1d(range(len(assembly)), assembly_idx['test'] )

        for mode,pair in pairs.items():
            print("Making {} pairs".format(mode))
            make_pairs([ assembly[idx] for idx in assembly_idx[mode] ],
                       step,frag_len,pairs[mode],n_examples=n_examples[mode])

    #######################
    ##### NN training #####
    #######################    
    
    model_output = "{}/{}.pth".format(io_path["out"],model_type)
    
    input_shapes = {
        'composition': [ sum([int(4**k / (1+train_args['rc'])) for k in kmer_list ]) ],
        'coverage': (int(frag_len/train_args['window_size']), n_samples)
    }

    model = initialize_model(model_type, input_shapes,
                             composition_args=nn_arch["composition"],
                             coverage_args=nn_arch["coverage"],
                             combination_args=nn_arch["combination"]
    )
    print("Model initialized")
    
    if not os.path.exists(model_output):
        train(model, pairs, model_output, model_type=model_type, **train_args, **filtered_inputs)
    else:
        checkpoint = torch.load(model_output)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

    ########################
    ###### Clustering ######
    ########################

    n_frags = cluster_args['n_frags']

    repr_outputs = {
        "composition": "{}/representation_compo_nf{}.h5".format(io_path["out"],n_frags),
        "coverage": "{}/representation_cover_nf{}.h5".format(io_path["out"],n_frags)
    }

    if not os.path.exists(repr_outputs['coverage']):
        save_repr_all(model,n_frags,frag_len,kmer_list,
                      train_args['rc'],train_args['window_size'],
                      outputs=repr_outputs,**filtered_inputs)

    cluster(model,repr_outputs,io_path["out"],**cluster_args)

    

if __name__ == '__main__':
    run()

