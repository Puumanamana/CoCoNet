import os

from Bio import SeqIO
import numpy as np
import torch

from experiment import Experiment

from preprocessing import format_assembly, bam_list_to_h5, filter_h5
from npy_fragmentation import make_pairs
from nn_training import initialize_model,train
from clustering import save_repr_all, cluster

def run(name):

    config = Experiment(name)
    
    #######################
    #### Preprocessing ####
    #######################

    if not os.path.exists(config.outdir):
        os.mkdir(config.outdir)

    if not os.path.exists(config.inputs['filtered']['fasta']):
        format_assembly(config.inputs['raw']['fasta'],
                        config.inputs['filtered']['fasta'],
                        min_length=config.min_ctg_len)

    if not os.path.exists(config.inputs['filtered']['coverage_h5']):
        if 'sim' in config.name:
            filter_h5(config.inputs['raw']['coverage_h5'],
                      config.inputs['filtered']['coverage_h5'],
                      min_length=config.min_ctg_len)
        else:
            bam_list_to_h5(config.inputs['filtered']['fasta'],
                           config.inputs['raw']['bam'],
                           output=config.inputs['filtered']['coverage_h5'])
        

    config.set_input_shapes()

    #######################
    #### Fragmentation ####
    #######################        

    if not os.path.exists(config.outputs['fragments']['train']):
        assembly = [ contig for contig in SeqIO.parse(config.inputs['filtered']['fasta'],"fasta") ]

        assembly_idx = { 'test': np.random.choice(len(assembly),int(0.1*len(assembly))) }
        assembly_idx['train'] = np.setdiff1d(range(len(assembly)), assembly_idx['test'] )

        for mode,pair in config.outputs['fragments'].items():
            print("Making {} pairs".format(mode))
            make_pairs([ assembly[idx] for idx in assembly_idx[mode] ],
                       config.step,config.fl,pair,n_examples=config.n_examples[mode])

    #######################
    ##### NN training #####
    #######################    

    # model = initialize_model(config.model_type, config.input_shapes,
    #                          composition_args=config.arch["composition"],
    #                          coverage_args=config.arch["coverage"],
    #                          combination_args=config.arch["combination"],
    #                          # pretrained_path="pretrained_model/composition_2-3-4.pth"
    # )
    model = initialize_model(config.model_type, config)

    print("Model initialized")
    
    if not os.path.exists(config.outputs['model']):
        train(model,config)

    checkpoint = torch.load(config.outputs['model'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    ########################
    ###### Clustering ######
    ########################

    if not os.path.exists(config.outputs['repr']['coverage']):
         save_repr_all(model,config)

    cluster(model,config.outputs['repr'],config.outdir,**config.clustering)    

if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        name = 'camisim_5'
    else:
        name = sys.argv[1]
    run(name)

