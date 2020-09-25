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
from coconet.clustering import make_pregraph, refine_clustering



def main(**kwargs):
    """
    Arisdakessian C., Nigro O., Stewart G., Poisson G., Belcaid M.
    CoCoNet: An Efficient Deep Learning Tool for Viral Metagenome Binning
    """

    args = parse_args()

    params = vars(args)

    if kwargs:
        params.update(kwargs)

    logger = setup_logger('CoCoNet', Path(params['output'], 'CoCoNet.log'), params['loglvl'])
    action = params.pop('action')

    prev_config = Path(args.output, 'config.yaml')

    if prev_config.is_file():
        cfg = Configuration.from_yaml(prev_config)
    else:
        cfg = Configuration()
    cfg.init_config(mkdir=True, **params)
    cfg.to_yaml()

    logger.info(f'Using {cfg.threads} threads')
    logger.info(f'Features: {", ".join(args.features)}')

    torch.set_num_threads(cfg.threads)

    if action == 'preprocess':
        preprocess(cfg)
    elif action == 'learn':
        make_train_test(cfg)
        learn(cfg)
        precompute_latent_repr(cfg)
    elif action == 'cluster':
        cluster(cfg)
    else:
        preprocess(cfg)
        make_train_test(cfg)
        learn(cfg)
        precompute_latent_repr(cfg)
        cluster(cfg)

def preprocess(cfg):
    """
    Preprocess assembly and coverage

    Args:
        cfg (coconet.core.config.Configuration)
    Returns:
        None
    """

    logger = setup_logger('preprocessing', cfg.io['log'], cfg.loglvl)

    composition = cfg.get_composition_feature()

    logger.info(f'Processing {composition.count("fasta"):,} contigs')
    composition.filter_by_length(output=cfg.io['filt_fasta'], min_length=cfg.min_ctg_len)
    logger.info((f'Length filter (L>{cfg.min_ctg_len} bp) -> '
                 f'{composition.count("filt_fasta"):,} contigs remaining'))

    if 'bam' in cfg.io or cfg.io['h5'].is_file():
        coverage = cfg.get_coverage_feature()

    indent = ' ' * 42 # For logging multiline-formatting
    if 'bam' in cfg.io:
        logger.info('Processing alignments and converting to h5 format')

        counts = coverage.to_h5(composition.get_valid_nucl_pos(), output=cfg.io['h5'],
                                tlen_range=cfg.tlen_range,
                                min_mapq=cfg.min_mapping_quality,
                                min_coverage=cfg.min_aln_coverage,
                                flag=cfg.flag)

        if counts is not None:
            bam_filtering_info = [
                'Coverage filtering summary:',
                f'{indent}- {counts[0]:,.0f} total reads',
                f'{indent}- {counts[2]:.1%} reads (mapped)',
                f'{indent}- {counts[1]:.1%} reads (primary alignments)',
                f'{indent}- {counts[3]:.1%} reads (mapq>{cfg.min_mapping_quality})',
                f'{indent}- {counts[4]:.1%} reads (coverage>{cfg.min_aln_coverage/100:.0%})',
                f'{indent}- {counts[5]:.1%} reads (flag & {cfg.flag} == 0)',
            ]

            if cfg.tlen_range is not None:
                bam_filtering_info.append(
                    '{}- {:.1%} reads ({}<=tlen<={})'
                    .format(indent, counts[-1], *cfg.tlen_range)
                )

            logger.info('\n'.join(bam_filtering_info))

    if cfg.io['h5'].is_file():
        # Make sure the contig IDs are the same for both coverage and composition. Take the intersection otherwise
        composition.synchronize(coverage, ['filt_fasta', 'h5'])
        # remove singletons
        coverage.find_singletons(output=cfg.io['singletons'], min_prevalence=cfg.min_prevalence)
        coverage.filter_by_ids(ids_file=cfg.io['singletons'])
        composition.filter_by_ids(ids_file=cfg.io['singletons'])
        logger.info((f'Prevalence filter (prevalence>={cfg.min_prevalence}) -> '
                     f'{composition.count("filt_fasta"):,} contigs remaining'))

def make_train_test(cfg):
    """
    - Split contigs into fragments
    - Make pairs of fragments such that we have:
       - n/2 positive examples (fragments from the same contig)
       - n/2 negative examples (fragments from different contigs)

    Args:
        cfg (coconet.core.config.Configuration)
    Returns:
        None
    """

    logger = setup_logger('learning', cfg.io['log'], cfg.loglvl)
    if not cfg.io['filt_fasta'].is_file():
        logger.warning('Input fasta file was not preprocessed. Using raw fasta instead.')
        cfg.io['filt_fasta'] = cfg.io['fasta']

    logger.info("Making train/test examples")
    composition = cfg.get_composition_feature()

    assembly = []
    for name, contig in composition.get_iterator('filt_fasta'):
        if len(contig) < cfg.min_ctg_len:
            logger.critical((
                f"Contig {name} is shorter than "
                f"the minimum length ({cfg.min_ctg_len}). Aborting"
            ))
            raise RuntimeError
        assembly.append((name, contig))

    n_ctg = len(assembly)
    n_ctg_for_test = max(2, int(cfg.test_ratio*n_ctg))

    assembly_idx = dict(test=np.random.choice(n_ctg, n_ctg_for_test))
    assembly_idx['train'] = np.setdiff1d(range(n_ctg), assembly_idx['test'])

    n_examples = dict(train=cfg.n_train, test=cfg.n_test)

    logger.info((
        f'Parameters: fragment_step={cfg.fragment_step}, '
        f'fragment_length={cfg.fragment_length}, '
        f'#examples (train)={n_examples["train"]}, '
        f'#examples (test)={n_examples["test"]}'
    ))
    for mode, pair_file in cfg.io['pairs'].items():
        make_pairs([assembly[idx] for idx in assembly_idx[mode]],
                   cfg.fragment_step,
                   cfg.fragment_length,
                   output=pair_file,
                   n_examples=n_examples[mode])

def learn(cfg):
    """
    Deep learning model

    Args:
        cfg (coconet.core.config.Configuration)
    Returns:
        None
    """

    logger = setup_logger('learning', cfg.io['log'], cfg.loglvl)

    model = load_model(cfg)
    model.train()

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
            logger.critical((
                f'{key} file not found at {path}. '
                'Did you run the preprocessing step with the {key} file?'
            ))
            raise FileNotFoundError

    if not all(f.is_file() for f in cfg.io['pairs'].values()):
        logger.critical(
            (f'Train/test sets not found at {path}.'
             'Did you delete the pair files?')
        )
        raise FileNotFoundError

    logger.info((
        f'Parameters: batch size={cfg.batch_size}, learning rate={cfg.learning_rate}, '
        f'kmer size={cfg.kmer}, cannonical={not cfg.no_rc}, '
        f'coverage smoothing=(wsize={cfg.wsize}, wstep={cfg.wstep}).'
    ))

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
        wstep=cfg.wstep
    )
    logger.info('Training finished')


def precompute_latent_repr(cfg):
    """
    Computes latent representation using the trained model.
    Saves output in .h5 file for each feature

    Args:
        cfg (coconet.core.config.Configuration)
    Returns:
        None
    """

    logger = setup_logger('learning', cfg.io['log'], cfg.loglvl)
    logger.info('Computing intermediate representation of composition and coverage features')

    model = load_model(cfg, from_checkpoint=True)

    save_repr_all(model,
                  fasta=cfg.io['filt_fasta'],
                  coverage=cfg.io['h5'],
                  output={feature: cfg.io['repr'][feature] for feature in cfg.features},
                  n_frags=cfg.n_frags,
                  frag_len=cfg.fragment_length,
                  rc=not cfg.no_rc,
                  wsize=cfg.wsize, wstep=cfg.wstep)
    return model

def cluster(cfg):
    """
    Make adjacency matrix and cluster contigs

    Args:
        cfg (coconet.core.config.Configuration)
    Returns:
        None
    """

    logger = setup_logger('clustering', cfg.io['log'], cfg.loglvl)

    full_cfg = Configuration.from_yaml('{}/config.yaml'.format(cfg.io['output']))
    model = load_model(full_cfg, from_checkpoint=True)

    if not all(x.is_file() for x in cfg.io['repr'].values()):
        logger.critical((
            f'Could not find the latent representations in {cfg.io["output"]}. '
            'Did you run coconet learn before?'
        ))
        raise FileNotFoundError

    features = cfg.get_features()

    logger.info('Pre-clustering contigs')
    logger.info(f'Parameters: alg={cfg.alg}, max neighbors={cfg.max_neighbors}, theta={cfg.theta}, gamma={cfg.gamma1}')
    make_pregraph(model, features, output=cfg.io['pre_graph'],
                  vote_threshold=cfg.vote_threshold,
                  max_neighbors=cfg.max_neighbors)

    logger.info('Refining graph')
    logger.info(f'Parameters: alg={cfg.alg}, theta={cfg.theta}, gamma={cfg.gamma2}, n_clusters={cfg.n_clusters}')
    refine_clustering(
        model,
        features,
        cfg.io['pre_graph'],
        singletons_file=cfg.io['singletons'],
        graph_file=cfg.io['graph'],
        assignments_file=cfg.io['assignments'],
        vote_threshold=cfg.vote_threshold,
        alg=cfg.alg,
        n_clusters=cfg.n_clusters,
        theta=cfg.theta,
        gamma1=cfg.gamma1,
        gamma2=cfg.gamma2
    )

if __name__ == '__main__':
    main()
