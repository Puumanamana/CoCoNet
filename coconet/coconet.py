#!/usr/bin/env python
'''
Root script to run CoCoNet

Inputs:
Outputs:
'''

from Bio import SeqIO
import numpy as np
import torch

from coconet.core.config import Configuration
from coconet.parser import parse_args
from coconet.fragmentation import make_pairs
from coconet.dl_util import initialize_model, load_model, train, save_repr_all
from coconet.clustering import make_pregraph, iterate_clustering

def main(**kwargs):
    '''
    Arisdakessian C., Nigro O., Stewart G., Poisson G., Belcaid M.
    CoCoNet: An Efficient Deep Learning Tool for Viral Metagenome Binning
    '''

    args = parse_args()
    params = vars(args)

    if kwargs:
        params.update(kwargs)

    if params['fasta'] is None and 'composition' in params['features']:
        raise ValueError("Could not find the .fasta file. Did you use the --fasta flag?")
    if (params['bam'] is None and params['h5'] is None) and 'coverage' in params['features']:
        raise ValueError("Could not find the .bam files. Did you use the --coverage flag?")

    cfg = Configuration()
    cfg.init_config(mkdir=True, **params)
    cfg.to_yaml()
    cfg.set_logging()

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

    cfg.logger.info(f'Preprocessing fasta and {cfg.cov_type} files')

    composition = cfg.get_composition_feature()
    composition.filter_by_length(output=cfg.io['filt_fasta'],
                                 min_length=cfg.min_ctg_len,
                                 logger=cfg.logger)
    composition.write_bed(output=cfg.io['bed'])

    cfg.logger.info('Converting bam coverage to hdf5')
    coverage = cfg.get_coverage_feature()
    coverage.filter_bams(outdir=cfg.io['output'], bed=cfg.io['bed'], threads=cfg.threads,
                         min_qual=cfg.min_mapping_quality, flag=cfg.flag,
                         fl_range=cfg.fl_range, logger=cfg.logger)
    coverage.bam_to_tsv(output=cfg.io['tsv'])
    coverage.tsv_to_h5(composition.get_valid_nucl_pos(), output=cfg.io['h5'])
    coverage.write_singletons(output=cfg.io['singletons'], min_prevalence=cfg.min_prevalence)

    if not coverage.path['h5'].is_file():
        cfg.logger.warning('Could not get coverage table. Is your input "bam" formatted?')

    
    composition.filter_by_ids(output=cfg.io['filt_fasta'],
                              ids_file=cfg.io['singletons'])

    summary = composition.summarize_filtering(singletons=cfg.io['singletons'])

    cfg.logger.info(f'Contig filtering: {summary}')

def make_train_test(cfg):
    '''
    - Split contigs into fragments
    - Make pairs of fragments such that we have:
       - n/2 positive examples (fragments from the same contig)
       - n/2 negative examples (fragments from different contigs)
    '''

    cfg.logger.info("Making train/test examples")
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
                   n_examples=n_examples[mode],
                   logger=cfg.logger)

def learn(cfg):
    '''
    Deep learning model
    '''

    torch.set_num_threads(cfg.threads)

    input_shapes = cfg.get_input_shapes()
    architecture = cfg.get_architecture()

    model = initialize_model('-'.join(cfg.features), input_shapes, architecture)
    device = list({p.device.type for p in model.parameters()})
    cfg.logger.info(f'Neural network training on {" and ".join(device)}')
    cfg.logger.debug(str(model))

    inputs = {}
    if 'composition' in cfg.features:
        inputs['fasta'] = cfg.io['filt_fasta']
    if 'coverage' in cfg.features:
        inputs['coverage'] = cfg.io['h5']
    
    train(
        model, **inputs,
        pairs=cfg.io['pairs'],
        test_output=cfg.io['nn_test'],
        output=cfg.io['model'],
        batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        kmer=cfg.kmer,
        rc=not cfg.no_rc,
        norm=cfg.norm,
        load_batch=cfg.load_batch,
        wsize=cfg.wsize,
        wstep=cfg.wstep,
        logger=cfg.logger
    )

    cfg.logger.info('Training finished')

    checkpoint = torch.load(cfg.io['model'])
    model.load_state_dict(checkpoint['state'])
    model.eval()

    cfg.logger.info('Computing intermediate representation of composition and coverage features')
    save_repr_all(model, fasta=cfg.io['filt_fasta'], coverage=cfg.io['h5'],
                  latent_composition=cfg.io['repr']['composition'],
                  latent_coverage=cfg.io['repr']['coverage'],                  
                  n_frags=cfg.n_frags,
                  frag_len=cfg.fragment_length,
                  rc=not cfg.no_rc,
                  wsize=cfg.wsize, wstep=cfg.wstep)
    return model

def cluster(cfg, force=False):
    '''
    Make adjacency matrix and cluster contigs
    '''

    torch.set_num_threads(cfg.threads)

    full_cfg = Configuration.from_yaml('{}/config.yaml'.format(cfg.io['output']))
    model = load_model(full_cfg)
    n_frags = full_cfg.n_frags

    cfg.logger.info('Pre-clustering contigs')
    make_pregraph(model, cfg.get_features(), output=cfg.io['pre_graph'],
                  n_frags=n_frags, max_neighbors=cfg.max_neighbors, force=force)

    cfg.logger.info('Refining graph')
    iterate_clustering(
        model, cfg.io['repr'], cfg.io['pre_graph'],
        singletons_file=cfg.io['singletons'],
        graph_file=cfg.io['graph'],
        assignments_file=cfg.io['assignments'],
        n_frags=n_frags,
        theta=cfg.theta,
        gamma1=cfg.gamma1, gamma2=cfg.gamma2,
        force=force
    )

if __name__ == '__main__':
    main()
