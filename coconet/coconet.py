#!/usr/bin/env python
'''
Root script to run CoCoNet

Inputs:
Outputs:
'''

import click
from Bio import SeqIO
import numpy as np
import torch

from coconet.config import Configuration
from coconet.parser_info import make_decorator, _HELP_MSG
from coconet.preprocessing import format_assembly, bam_list_to_h5, filter_h5
from coconet.fragmentation import make_pairs
from coconet.dl_util import initialize_model, load_model, train, save_repr_all
from coconet.clustering import fill_adjacency_matrix, iterate_clustering

CONTEXT_SETTINGS = {'help_option_names': ['-h', '--help']}
PASS_CONTEXT = click.make_pass_decorator(Configuration, ensure=True)

class NaturalOrderGroup(click.Group):
    '''
    Needed to make order the subcommands in order of appearance in the script
    '''
    def list_commands(self, ctx):
        return self.commands.keys()

@click.group(cls=NaturalOrderGroup, context_settings=CONTEXT_SETTINGS)
def main():
    '''
    Arisdakessian C., Nigro O., Stewart G., Poisson G., Belcaid M.
    CoCoNet: An Efficient Deep Learning Tool for Viral Metagenome Binning
    '''

    print('''
    ############################################
    #### Starting CoCoNet binning algorithm ####
    ############################################
    ''')

@main.command(help=_HELP_MSG['preprocess'])
@make_decorator('io', 'general', 'preproc')
@PASS_CONTEXT
def preprocess(cfg, **kwargs):
    '''
    Preprocess assembly and coverage
    '''

    cfg.init_config(mkdir=True, **kwargs)
    cfg.to_yaml()

    print('Preprocessing fasta and {} files'.format(cfg.cov_type))

    if cfg.cov_type == '.bam':
        _ = format_assembly(cfg.io['fasta'],
                            output=cfg.io['filt_fasta'],
                            min_length=cfg.min_ctg_len)

        bam_list_to_h5(
            cfg.io['filt_fasta'], cfg.io['coverage_bam'],
            output=cfg.io['filt_h5'],
            singleton_file=cfg.io['singletons'],
            threads=cfg.threads, min_prevalence=cfg.min_prevalence, tmp_dir=cfg.io['tmp_dir'],
            min_qual=cfg.min_mapping_quality, flag=cfg.flag, fl_range=cfg.fl_range
        )

    # Input is h5 formatted
    else:
        filter_h5(cfg.io['fasta'], cfg.io['coverage_h5'],
                  filt_fasta=cfg.io['filt_fasta'], filt_h5=cfg.io['filt_h5'],
                  singleton_file=cfg.io['singletons'],
                  min_length=cfg.min_ctg_len, min_prevalence=cfg.min_prevalence)

@main.command(help=_HELP_MSG['make_train_test'])
@make_decorator('io', 'general', 'frag')
@PASS_CONTEXT
def make_train_test(cfg, **kwargs):
    '''
    - Split contigs into fragments
    - Make pairs of fragments such that we have:
       - n/2 positive examples (fragments from the same contig)
       - n/2 negative examples (fragments from different contigs)
    '''

    cfg.init_config(mkdir=True, **kwargs)
    cfg.to_yaml()

    print("Making train/test examples")
    assembly = [contig for contig in SeqIO.parse(cfg.io['filt_fasta'], 'fasta')]

    assembly_idx = {'test': np.random.choice(len(assembly), int(0.1*len(assembly)))}
    assembly_idx['train'] = np.setdiff1d(range(len(assembly)), assembly_idx['test'])

    n_examples = {'train': cfg.n_train, 'test': cfg.n_test}

    for mode, pair_file in cfg.io['pairs'].items():
        make_pairs([assembly[idx] for idx in assembly_idx[mode]],
                   cfg.fragment_step,
                   cfg.fragment_length,
                   output=pair_file,
                   n_examples=n_examples[mode])

@main.command(help=_HELP_MSG['learn'])
@make_decorator('io', 'general', 'dl')
@PASS_CONTEXT
def learn(cfg, **kwargs):
    '''
    Deep learning model
    '''

    cfg.init_config(mkdir=True, **kwargs)
    cfg.to_yaml()

    torch.set_num_threads(cfg.threads)

    input_shapes = cfg.get_input_shapes()
    architecture = cfg.get_architecture()

    print('Neural network training')

    model = initialize_model('CoCoNet', input_shapes, architecture)
    print(model)

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

@main.command(help=_HELP_MSG['cluster'])
@make_decorator('io', 'general', 'cluster')
@PASS_CONTEXT
def cluster(cfg, **kwargs):
    '''
    Make adjacency matrix and cluster contigs
    '''

    cfg.init_config(mkdir=True, **kwargs)
    cfg.to_yaml()

    torch.set_num_threads(cfg.threads)

    full_cfg = Configuration.from_yaml('{}/config.yaml'.format(cfg.io['output']))
    model = load_model(full_cfg)

    print('Computing adjacency matrix')
    fill_adjacency_matrix(model, cfg.io['repr'], output=cfg.io['adjacency_matrix'],
                          n_frags=cfg.n_frags, max_neighbors=cfg.max_neighbors)

    print('Clustering contigs')
    iterate_clustering(
        model, cfg.io['repr'], cfg.io['adjacency_matrix'],
        singletons_file=cfg.io['singletons'],
        refined_adj_mat_file=cfg.io['refined_adjacency_matrix'],
        assignments_file=cfg.io['assignments'],
        refined_assignments_file=cfg.io['refined_assignments'],
        n_frags=cfg.n_frags,
        hits_threshold=cfg.hits_thresh,
        gamma1=cfg.gamma1,
        gamma2=cfg.gamma2,
        max_neighbors=cfg.max_neighs,
    )

@main.command(help=_HELP_MSG['run'])
@make_decorator('cluster', 'dl', 'frag', 'preproc', 'general', 'io')
@click.pass_context
def run(context, **_kwargs):
    '''
    Run complete pipeline
    '''

    context.forward(preprocess)
    context.forward(make_train_test)
    context.forward(learn)
    context.forward(cluster)

if __name__ == '__main__':
    main()
