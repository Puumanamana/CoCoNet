'''
Root script to run CoCoNet

Inputs:
Outputs:
'''

import click
from Bio import SeqIO
import numpy as np
import torch

from config import Configuration
from parser_info import make_decorator
from preprocessing import format_assembly, bam_list_to_h5, filter_h5
from fragmentation import make_pairs
from dl_util import initialize_model, train, save_repr_all
from clustering import fill_adjacency_matrix, iterate_clustering

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
pass_context = click.make_pass_decorator(Configuration, ensure=True)

@click.group(context_settings=CONTEXT_SETTINGS)
def main():
    '''
    - Greeter
    '''

    print('#### Starting CoCoNet ####\n')

@main.command(help='Preprocess the contig coverage')
@make_decorator('io', 'general', 'preproc')
@pass_context
def preprocessing(cfg, **kwargs):
    '''
    Preprocessing steps
    '''

    cfg.init_config(**kwargs)
    cfg.to_yaml()

    if cfg.cov_type == '.bam':
        format_assembly(cfg.io['fasta'], cfg.io['filt_fasta'], min_length=cfg.min_ctg_len)

        bam_list_to_h5(
            cfg.io['filt_fasta'], cfg.coverage_bam,
            output=cfg.io['filt_h5'],
            singleton_file=cfg.io['singletons'],
            threads=cfg.threads, min_prevalence=cfg.min_prevalence, tmp_dir=cfg.io['tmp_dir'],
            min_qual=cfg.min_mapping_quality, flag=cfg.flag, fl_range=cfg.fl_range
        )

    # Input is h5 formatted
    filter_h5(cfg.io['fasta'], cfg.io['coverage_h5'],
              filt_fasta=cfg.io['filt_fasta'], filt_h5=cfg.io['filt_h5'],
              singleton_file=cfg.io['singletons'],
              min_length=cfg.min_ctg_len, min_prevalence=cfg.min_prevalence)

@main.command(help='Fragmentation tool to split the contigs')
@make_decorator('io', 'general', 'frag')
@pass_context
def fragmentation(cfg, **kwargs):
    '''
    Fragmentation steps
    '''

    cfg.init_config(**kwargs)
    cfg.to_yaml()

    assembly = [contig for contig in SeqIO.parse(cfg.io['filt_fasta'], 'fasta')]

    assembly_idx = {'test': np.random.choice(len(assembly), int(0.1*len(assembly)))}
    assembly_idx['train'] = np.setdiff1d(range(len(assembly)), assembly_idx['test'])

    n_examples = {'train': cfg.n_train, 'test': cfg.n_test}

    for mode, pair_file in cfg.io['pairs'].items():
        print("Making {} pairs".format(mode))
        make_pairs([assembly[idx] for idx in assembly_idx[mode]],
                   cfg.fragment_step,
                   cfg.fragment_length,
                   output=pair_file,
                   n_examples=n_examples[mode])

@main.command(help='Train neural network')
@make_decorator('io', 'general', 'dl')
@pass_context
def learn(cfg, **kwargs):
    '''
    Deep learning model
    '''

    cfg.init_config(**kwargs)
    cfg.to_yaml()

    torch.set_num_threads(cfg.threads)

    input_shapes = cfg.get_input_shapes()
    architecture = cfg.get_architecture()

    model = initialize_model('CoCoNet', input_shapes, architecture)

    train(
        model,
        cfg.io['filt_fasta'],
        cfg.io['filt_h5'],
        cfg.io['pairs'],
        cfg.io['nn_test'],
        output=cfg.io['model'],
        batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        kmer=cfg.kmer,
        rc=not cfg.no_rc,
        norm=cfg.norm,
        load_batch=cfg.load_batch,
        wsize=cfg.wsize,
        wstep=cfg.wstep
    )

    checkpoint = torch.load(cfg.io['model'])
    model.load_state_dict(checkpoint['state'])
    model.eval()

    save_repr_all(model, cfg.io['filt_fasta'], cfg.io['filt_h5'],
                  output=cfg.io['repr'],
                  n_frags=cfg.n_frags,
                  frag_len=cfg.fragment_length,
                  rc=not cfg.no_rc,
                  wsize=cfg.wsize, wstep=cfg.wstep)
    return model

@main.command(help='Cluster contigs')
@make_decorator('io', 'general', 'dl', 'cluster')
@pass_context
def cluster(cfg, **kwargs):
    '''
    Make adjacency matrix and cluster contigs
    '''

    cfg.init_config(**kwargs)
    cfg.to_yaml()

    torch.set_num_threads(cfg.threads)

    input_shapes = cfg.get_input_shapes()
    architecture = cfg.get_architecture()
    model = initialize_model('CoCoNet', input_shapes, architecture)

    checkpoint = torch.load(cfg.io['model'])
    model.load_state_dict(checkpoint['state'])
    model.eval()

    fill_adjacency_matrix(model, cfg.io['repr'], output=cfg.io['adjacency_matrix'],
                          n_frags=cfg.n_frags, max_neighbors=cfg.max_neighbors)


    iterate_clustering(
        model,
        latent=cfg.io['repr'],
        singletons=cfg.io['singletons'],
        adj_mat=cfg.io['adjacency_matrix'],
        refined_adj_mat=cfg.io['refined_adjacency_matrix'],
        assignments=cfg.io['assignments'],
        refined_assignments=cfg.io['refined_assignments'],
        n_frags=cfg.n_frags,
        hits_threshold=cfg.hits_threshold,
        gamma1=cfg.gamma1,
        gamma2=cfg.gamma2,
        max_neighbors=cfg.max_neighbors,
    )

@main.command(help='Run complete pipeline')
@make_decorator('io', 'general', 'preproc', 'frag', 'dl', 'cluster')
@click.pass_context
def run(context, **kwargs):
    '''
    Run complete pipeline
    '''

    context.forward(preprocessing)
    context.forward(fragmentation)
    context.forward(learn)
    context.forward(cluster)

if __name__ == '__main__':
    main()
