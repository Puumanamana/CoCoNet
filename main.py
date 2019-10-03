'''
Root script to run CoCoNet

Inputs:
Outputs:
'''

import os
import argparse

from Bio import SeqIO
import numpy as np
import torch

from experiment import Experiment

from preprocessing import format_assembly, bam_list_to_h5, filter_h5
from fragmentation import make_pairs
from nn_training import initialize_model, train, test_summary
from clustering import save_repr_all, fill_adjacency_matrix, iterate_clustering

def parse_args():
    '''
    Parse arguments to run CoCoNet
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='')

    args = parser.parse_args()

    return args

def run():
    '''
    CoCoNet runner
    '''

    args = parse_args()

    config = Experiment(args.name)

    torch.set_num_threads(config.threads)

    #######################
    #### Preprocessing ####
    #######################

    if not os.path.exists(config.inputs['filtered']['fasta']):
        format_assembly(config.inputs['raw']['fasta'],
                        config.inputs['filtered']['fasta'],
                        min_length=config.min_ctg_len)

    if not os.path.exists(config.inputs['filtered']['coverage_h5']):
        if 'sim' in config.name or config.name.replace('_', '').isdecimal():
            filter_h5(config.inputs['raw']['coverage_h5'],
                      config.inputs['filtered']['coverage_h5'],
                      min_length=config.min_ctg_len)
        else:
            bam_list_to_h5(fasta=config.inputs['filtered']['fasta'],
                           coverage_bam=config.inputs['raw']['bam'],
                           output=config.inputs['filtered']['coverage_h5'],
                           threads=config.threads,
                           **config.bam_processing)

    config.set_input_shapes()

    #######################
    #### Fragmentation ####
    #######################

    if not os.path.exists(config.outputs['fragments']['train']):
        assembly = [contig for contig in SeqIO.parse(config.inputs['filtered']['fasta'], "fasta")]

        assembly_idx = {'test': np.random.choice(len(assembly), int(0.1*len(assembly)))}
        assembly_idx['train'] = np.setdiff1d(range(len(assembly)), assembly_idx['test'])

        for mode, pair in config.outputs['fragments'].items():
            print("Making {} pairs".format(mode))
            make_pairs([assembly[idx] for idx in assembly_idx[mode]],
                       config.step, config.fl, pair, n_examples=config.n_examples[mode])

    #######################
    ##### NN training #####
    #######################

    model = initialize_model(config.model_type, config)

    if not os.path.exists(config.outputs['net']['model']):
        train(model, config)

    checkpoint = torch.load(config.outputs['net']['model'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    _ = test_summary(model, config=config)

    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # sns.kdeplot(output['combined'][:, 0].detach().numpy())
    # plt.show()
    # import ipdb
    # ipdb.set_trace()

    print('Network loaded')

    ########################
    ###### Clustering ######
    ########################

    if not os.path.exists(config.outputs['repr']['coverage']):
        save_repr_all(model, config)

    fill_adjacency_matrix(model, config)

    if not os.path.exists(config.outputs['clustering']['refined_assignments']):
        iterate_clustering(model, config)

if __name__ == '__main__':
    run()
