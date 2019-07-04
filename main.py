import os
from Bio import SeqIO
import numpy as np
import h5py
import torch

from config import io_path
from config import frag_len,step,kmer,model_type
from config import nn_arch, train_args
from config import cluster_args

from fragmentation import make_pairs
from nn_training import initialize_model,train
from clustering import save_repr_all, cluster

def run():

    # format_assembly()

    input_files = { "fasta": "{}/assembly.fasta".format(io_path["in"]),
                    "coverage_h5": "{}/coverage_contigs.h5".format(io_path["in"])}

    h5_cov = h5py.File(input_files['coverage_h5'],'r')
    n_samples = h5_cov.get(list(h5_cov.keys())[0]).shape[0]

    #######################
    #### Fragmentation ####
    #######################        

    pairs = {
        "test": "{}/pairs_test.csv".format(io_path["out"]),        
        "train": "{}/pairs_train.csv".format(io_path["out"])
    }

    if not os.path.exists(pairs["train"]):
        assembly = [ contig for contig in SeqIO.parse(input_files["fasta"],"fasta") ]

        assembly_idx = { 'test': np.random.choice(len(assembly),int(0.1*len(assembly))) }
        assembly_idx['train'] = np.setdiff1d(range(len(assembly)), assembly_idx['test'] )

        n_examples = { 'train': int(1e6), 'test': int(2e4) }

        for mode,pair in pairs.items():
            print("Making {} pairs".format(mode))
            make_pairs([ assembly[idx] for idx in assembly_idx[mode] ],
                       step,frag_len,pairs[mode],n_examples=n_examples[mode])

    #######################
    ##### NN training #####
    #######################    
    
    model_output = "{}/{}.pth".format(io_path["out"],model_type)
    
    input_shapes = {
        'composition': int(4**kmer / (1+train_args['rc'])),
        'coverage': (int(frag_len/train_args['window_size']), n_samples)
    }

    model = initialize_model(model_type, input_shapes,
                             composition_args=nn_arch["composition"],
                             coverage_args=nn_arch["coverage"],
                             combination_args=nn_arch["combination"]
    )
    
    if not os.path.exists(model_output):
        train(model, pairs, model_output, model_type=model_type, **train_args, **input_files)
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
        save_repr_all(model,n_frags,frag_len,kmer,train_args['window_size'],
                      outputs=repr_outputs,**input_files)

    cluster(model,repr_outputs,io_path["out"],**cluster_args)

    

if __name__ == '__main__':
    run()

