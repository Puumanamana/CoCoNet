#!/usr/bin/env python
'''
Root script to run CoCoNet
'''

from pathlib import Path
import numpy as np
import torch

from coconet.log import setup_logger
from coconet.core.config import Configuration
from coconet.parser import parse_args
from coconet.fragmentation import make_pairs
from coconet.dl_util import load_model, train, save_repr_all
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

    setup_logger('CoCoNet', Path(params['output'], 'CoCoNet.log'), params['loglvl'])
    action = params.pop('action')

    cfg = Configuration()
    cfg.init_config(mkdir=True, **params)
    cfg.to_yaml()

    if action == 'preprocess':
        preprocess(cfg)
    elif action == 'learn':
        make_train_test(cfg)
        learn(cfg)
    elif action == 'cluster':
        cluster(cfg)
    else:
        preprocess(cfg)
        make_train_test(cfg)
        learn(cfg)
        cluster(cfg)

def preprocess(cfg):
    '''
    Preprocess assembly and coverage
    '''

    logger = setup_logger('preprocessing', cfg.io['log'], cfg.loglvl)

    if cfg.min_ctg_len < 0:
        cfg.min_ctg_len = 2 * cfg.fragment_length

    logger.info(f'Filtering fasta')
    composition = cfg.get_composition_feature()
    composition.filter_by_length(output=cfg.io['filt_fasta'], min_length=cfg.min_ctg_len)

    if 'bam' in cfg.io:
        logger.info('Converting bam coverage to hdf5')
        coverage = cfg.get_coverage_feature()
        # TO DO: take parameters into account the flag
        coverage.to_h5(composition.get_valid_nucl_pos(), output=cfg.io['h5'],
                       tlen_range=cfg.fl_range,
                       min_mapq=cfg.min_mapping_quality,
                       min_coverage=cfg.min_aln_coverage)

    if cfg.io['h5'].is_file():
        coverage.write_singletons(output=cfg.io['singletons'], min_prevalence=cfg.min_prevalence)
        composition.filter_by_ids(output=cfg.io['filt_fasta'], ids_file=cfg.io['singletons'])

    summary = composition.summarize_filtering(singletons=cfg.io['singletons'])
    logger.info(f'Contig filtering - {summary}')


def make_train_test(cfg):
    '''
    - Split contigs into fragments
    - Make pairs of fragments such that we have:
       - n/2 positive examples (fragments from the same contig)
       - n/2 negative examples (fragments from different contigs)
    '''

    logger = setup_logger('learning', cfg.io['log'], cfg.loglvl)
    if not cfg.io['filt_fasta'].is_file():
        logger.warning('Input fasta file was not preprocessed. Using raw fasta instead.')
        cfg.io['filt_fasta'] = cfg.io['fasta']

    logger.info("Making train/test examples")
    composition = cfg.get_composition_feature()

    assembly = [x for x in composition.get_iterator('filt_fasta')]
    n_ctg = len(assembly)

    n_ctg_for_test = max(2, int(cfg.test_ratio*n_ctg))

    assembly_idx = {'test': np.random.choice(n_ctg, n_ctg_for_test)}
    assembly_idx['train'] = np.setdiff1d(range(n_ctg), assembly_idx['test'])

    n_examples = {'train': cfg.n_train, 'test': cfg.n_test}

    for mode, pair_file in cfg.io['pairs'].items():
        make_pairs([assembly[idx] for idx in assembly_idx[mode]],
                   cfg.fragment_step,
                   cfg.fragment_length,
                   output=pair_file,
                   n_examples=n_examples[mode],
                   logger=logger)

def learn(cfg):
    '''
    Deep learning model
    '''

    logger = setup_logger('learning', cfg.io['log'], cfg.loglvl)

    torch.set_num_threads(cfg.threads)

    model = load_model(cfg)

    device = list({p.device.type for p in model.parameters()})
    logger.info(f'Neural network training on {" and ".join(device)}')
    logger.debug(str(model))

    inputs = {}
    if 'composition' in cfg.features:
        inputs['fasta'] = cfg.io['filt_fasta']
    if 'coverage' in cfg.features:
        inputs['coverage'] = cfg.io['h5']

    for (key, path) in inputs.items():
        if not path.is_file():
            logger.critical(
                (f'{key} file not found at {path}. '
                 'Did you run the preprocessing step with the {key} file?')
            )
            raise FileNotFoundError

    if not all(f.is_file() for f in cfg.io['pairs'].values()):
        logger.critical(
            (f'Train/test sets not found at {path}.'
             'Did you delete the pair files?')
        )
        raise FileNotFoundError

    model = train(
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
        wstep=cfg.wstep
    )
    logger.info('Training finished')

    logger.info('Computing intermediate representation of composition and coverage features')
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

    logger = setup_logger('clustering', cfg.io['log'], cfg.loglvl)

    torch.set_num_threads(cfg.threads)

    full_cfg = Configuration.from_yaml('{}/config.yaml'.format(cfg.io['output']))
    model = load_model(full_cfg)
    n_frags = full_cfg.n_frags

    if not all(x.is_file() for x in cfg.io['repr'].values()):
        logger.critical(
            (f'Could not find the latent representations in {cfg.io["output"]}. '
             'Did you run coconet learn before?')
        )

    features = cfg.get_features()

    logger.info('Pre-clustering contigs')
    make_pregraph(model, features, output=cfg.io['pre_graph'],
                  n_frags=n_frags, max_neighbors=cfg.max_neighbors, force=force)

    logger.info('Refining graph')
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
