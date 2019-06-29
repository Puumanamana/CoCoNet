import os
from Bio import SeqIO
import numpy as np
import h5py

from config import io_path
from config import frag_len,step,n_frags,kmer,add_rc
from config import nn_arch, train_args

from fragmentation import make_pairs
from nn_training import initialize_model,train

def run():

    # format_assembly()

    input_files = { "fasta": "{}/assembly.fasta".format(io_path["in"]),
                    "coverage_h5": "{}/coverage_contigs.h5".format(io_path["in"])}

    h5_cov = h5py.File(input_files['coverage_h5'],'r')
    n_samples = h5_cov.get(list(h5_cov.keys())[0]).shape[0]

    pairs = {
        "test": "{}/pairs_test.csv".format(io_path["out"]),        
        "train": "{}/pairs_train.csv".format(io_path["out"])
    }

    if not os.path.exists(pairs["train"]):
        assembly = [ contig for contig in SeqIO.parse(input_files["fasta"],"fasta") ]

        assembly_idx = { 'test': np.random.choice(len(assembly),int(0.05*len(assembly))) }
        assembly_idx['train'] = np.setdiff1d(range(len(assembly)), assembly_idx['test'] )

        n_examples = { 'train': int(1e6), 'test': int(1e4) }

        for mode,pair in pairs.items():
            print("Making {} pairs".format(mode))
            make_pairs([ assembly[idx] for idx in assembly_idx[mode] ],
                       n_frags,step,frag_len,pairs[mode],n_examples=n_examples[mode])

    model_output = "{}/CoCoNet.pth".format(io_path["out"])
    
    if not os.path.exists(model_output):

        input_shapes = {
            'composition': int(4**kmer / (1+add_rc)),
            'coverage': (int(frag_len/train_args['window_size']), n_samples)
        }
        
        model = initialize_model("CoCoNet", input_shapes,
                                 composition_args=nn_arch["composition"],
                                 coverage_args=nn_arch["coverage"],
                                 combination_args=nn_arch["combination"]
        )

        train(model, pairs, model_output, **train_args, **input_files)

if __name__ == '__main__':
    run()

