#!/usr/bin/env python
'''
Root script to run CoCoNet

Inputs:
Outputs:
'''

from Bio import SeqIO
import numpy as np
import torch

from coconet.config import Configuration
from coconet.parser import parse_args
from coconet.preprocessing import format_assembly, bam_list_to_h5, filter_h5
from coconet.fragmentation import make_pairs
from coconet.dl_util import initialize_model, load_model, train, save_repr_all
from coconet.clustering import make_pregraph, iterate_clustering

def main(**kwargs):
    '''
    Arisdakessian C., Nigro O., Stewart G., Poisson G., Belcaid M.
    CoCoNet: An Efficient Deep Learning Tool for Viral Metagenome Binning
    '''

    print('''
    ############################################
    #### Starting CoCoNet binning algorithm ####
    ############################################
    ''')

    args = parse_args()
    params = vars(args)

    if kwargs:
        params.update(kwargs)

    if params['fasta'] is None:
        raise ValueError("Could not find the .fasta file. Did you use the --fasta flag?")
    if params['coverage'] is None:
        raise ValueError("Could not find the .bam files. Did you use the --coverage flag?")

    cfg = Configuration()
    cfg.init_config(mkdir=True, **params)
    cfg.to_yaml()

    preprocess(cfg)
    make_train_test(cfg)
    learn(cfg)
    cluster(cfg)

def preprocess(cfg):
    '''
    Preprocess assembly and coverage
    '''

    if cfg.min_ctg_len < 0:
        cfg.min_ctg_len = 2 * cfg.fragment_length

    print('Preprocessing fasta and {} files'.format(cfg.cov_type))

    if cfg.cov_type == '.bam':
        format_assembly(cfg.io['fasta'],
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

def make_train_test(cfg):
    '''
    - Split contigs into fragments
    - Make pairs of fragments such that we have:
       - n/2 positive examples (fragments from the same contig)
       - n/2 negative examples (fragments from different contigs)
    '''

    print("Making train/test examples")
    assembly = [contig for contig in SeqIO.parse(cfg.io['filt_fasta'], 'fasta')]

    n_ctg_for_test = max(2, int(cfg.test_ratio*len(assembly)))

    assembly_idx = {'test': np.random.choice(len(assembly), n_ctg_for_test)}
    assembly_idx['train'] = np.setdiff1d(range(len(assembly)), assembly_idx['test'])

    n_examples = {'train': cfg.n_train, 'test': cfg.n_test}

    for mode, pair_file in cfg.io['pairs'].items():
        make_pairs([assembly[idx] for idx in assembly_idx[mode]],
                   cfg.fragment_step,
                   cfg.fragment_length,
                   output=pair_file,
                   n_examples=n_examples[mode])

def learn(cfg):
    '''
    Deep learning model
    '''

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

def cluster(cfg):
    '''
    Make adjacency matrix and cluster contigs
    '''

    torch.set_num_threads(cfg.threads)

    full_cfg = Configuration.from_yaml('{}/config.yaml'.format(cfg.io['output']))
    model = load_model(full_cfg)
    n_frags = full_cfg.n_frags

    print('Computing pre-graph')
    make_pregraph(model, cfg.io['repr'], output=cfg.io['pre_graph'],
                  n_frags=n_frags, max_neighbors=cfg.max_neighbors)

    print('Finalizing graph')
    iterate_clustering(
        model, cfg.io['repr'], cfg.io['pre_graph'],
        singletons_file=cfg.io['singletons'],
        graph_file=cfg.io['graph'],
        assignments_file=cfg.io['assignments'],
        n_frags=n_frags,
        theta=cfg.theta,
        gamma1=cfg.gamma1, gamma2=cfg.gamma2
    )

if __name__ == '__main__':
    main()
