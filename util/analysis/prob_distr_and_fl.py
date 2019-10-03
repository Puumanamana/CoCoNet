import os
from itertools import product
from functools import partial
import argparse
from multiprocessing.pool import Pool

import h5py
import numpy as np
import pandas as pd
from Bio import SeqIO
import torch

import seaborn as sns
import matplotlib.pyplot as plt

from clustering_errors import load_and_process_assignments, load_and_process_adj, PARENT_DIR

from experiment import Experiment
from nn_training import initialize_model
from tools import get_kmer_frequency, get_coverage, avg_window

PARAMS_1 = {'minlen': 2000, 'maxlen': 100000, 'frag_len': [30]}
PARAMS_2 = {'minlen': 2000, 'maxlen': 100000, 'frag_len': [30]}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_genomes', type=int, default=1000)
    parser.add_argument('--coverage', type=int, default=5)
    parser.add_argument('--n_samples', type=int, default=5)
    parser.add_argument('--name', type=str, default='')    

    args = parser.parse_args()

    return args

def load_model(config):
    model = initialize_model(config.model_type, config)
    checkpoint = torch.load(config.outputs['net']['model'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model

def get_ctg_compo_feature(ctg, n_frags, config, pool_obj):
    parser = SeqIO.parse(config.inputs['filtered']['fasta'], 'fasta')
    dna_seq = [str(x.seq) for x in parser if x.id == ctg][0]

    get_kmer_frequency_with_args = partial(
        get_kmer_frequency, kmer_list=config.kmer_list, rc=config.rc, norm=config.norm
    )

    step = int((len(dna_seq)-config.fl) / n_frags)
    dna_fragments = [dna_seq[step*i:step*i+config.fl] for i in range(n_frags)]

    x_compo = torch.from_numpy(np.array(
        pool_obj.map(get_kmer_frequency_with_args, dna_fragments)
    ).astype(np.float32))

    return x_compo

def get_ctg_cover_feature(ctg, n_frags, config):

    _, cov_len = config.input_shapes['coverage']
    cov_h5 = h5py.File(config.inputs['filtered']['coverage_h5'])

    step = int((cov_len-config.fl) / n_frags)
    fragment_boundaries = [(step*i, step*i+config.fl) for i in range(n_frags)]

    fragment_slices = np.array([np.arange(start, stop) for (start, stop) in fragment_boundaries])

    coverage_genome = np.array(cov_h5.get(ctg)[:]).astype(np.float32)[:, fragment_slices]
    coverage_genome = np.swapaxes(coverage_genome, 1, 0)

    x_cover = torch.from_numpy(
        np.apply_along_axis(
            lambda x: avg_window(x, config.wsize, config.wstep), 2, coverage_genome
        ).astype(np.float32))

    return x_cover

def build_outcome_matrix(adj, config, contigs,
                         pos_thresh=700, neg_thresh=600,
                         ex_per_type=100000):

    ctg_lengths = pd.Series({seq.id: len(seq.seq) for seq in
                             SeqIO.parse(config.inputs['filtered']['fasta'], 'fasta')})

    outcomes = np.recarray([4*ex_per_type],
                           dtype=[('ctg1', '<U128'), ('ctg2', '<U128'),
                                  ('l1', 'uint32'), ('l2', 'uint32'),
                                  ('type', '<U8')])

    with np.errstate(invalid='ignore'):
        cases = [
            ('TN', (np.abs(adj) < neg_thresh) & (adj < 0)),
            ('FP', (np.abs(adj) > pos_thresh) & (adj < 0)),
            ('FN', (np.abs(adj) < neg_thresh) & (adj > 0)),
            ('TP', (np.abs(adj) > pos_thresh) & (adj > 0)),
        ]

    for i, (out_type, condition) in enumerate(cases):
        data = np.argwhere(condition)
        indices = np.random.choice(data.shape[0], ex_per_type, replace=True)
        outcome_sub = outcomes[i*ex_per_type: (i+1)*ex_per_type]

        outcome_sub['ctg1'] = contigs[data[indices]][:, 0]
        outcome_sub['ctg2'] = contigs[data[indices]][:, 1]
        outcome_sub['l1'] = ctg_lengths.loc[outcome_sub['ctg1']].values
        outcome_sub['l2'] = ctg_lengths.loc[outcome_sub['ctg2']].values
        outcome_sub['type'] = out_type

    return outcomes

def pairwise_comp(config, net, outcomes, errType):
    '''
    Compare 2 contig's fragments
    by running the neural network for each fragment combinations
    between the contig pair
    '''

    pool = Pool(20)

    outcomes_filt = outcomes[
        (outcomes['l1'] > PARAMS_1['minlen'])
        & (outcomes['l1'] < PARAMS_1['maxlen'])
        & (outcomes['l2'] > PARAMS_2['minlen'])
        & (outcomes['l2'] < PARAMS_2['maxlen'])
        & (outcomes['type'] == errType)
    ]

    ctg1, ctg2, len_1, len_2, _ = outcomes_filt[np.random.choice(len(outcomes_filt))]
    print(ctg1, len_1, ctg2, len_2)

    probs_df_all = []
    for n_frags_ in product(PARAMS_1['frag_len'], PARAMS_2['frag_len']):
        indices_ = (np.repeat(np.arange(n_frags_[0]), n_frags_[1]),
                    np.tile(np.arange(n_frags_[1]), n_frags_[0]))

        x_compo = [get_ctg_compo_feature(ctg, n_frags, config, pool)[indices]
                   for ctg, n_frags, indices in zip([ctg1, ctg2], n_frags_, indices_)]

        x_cover = [get_ctg_cover_feature(ctg, n_frags, config)[indices]
                   for ctg, n_frags, indices in zip([ctg1, ctg2], n_frags_, indices_)]

        probs_df = pd.DataFrame({k: p.detach().numpy()[:, 0]
                                 for k, p in net(x_compo, x_cover).items()})
        probs_df['n_frags_1'] = n_frags_[0]
        probs_df['n_frags_2'] = n_frags_[1]
        probs_df_all.append(probs_df)

    pool.close()

    return ctg1, ctg2, pd.concat(probs_df_all).melt(id_vars=['n_frags_1', 'n_frags_2'])


def main():
    params = parse_args()

    if params.name == "":
        dataset = "{}_{}_{}".format(params.n_genomes, params.coverage, params.n_samples)
    else:
        dataset = params.name

    cfg = Experiment(dataset, root_dir=PARENT_DIR)
    cfg.set_input_shapes()

    assignments, _, mask = load_and_process_assignments(cfg)
    adj_mat = load_and_process_adj(cfg, assignments.truth.values, mask)
    results = build_outcome_matrix(adj_mat, cfg, assignments.contigs.values)
    model = load_model(cfg)

    choices = ['TP', 'FP', 'TN', 'FN', 'I']

    while 1:
        answer = input('Choose one: {}? '.format('/'.join(choices)+'/[Q]')).strip().upper()

        if answer == 'I':
            import ipdb
            ipdb.set_trace()

        if answer not in choices:
            break

        print('Displaying {}'.format(answer))

        ctg1, ctg2, probs = pairwise_comp(cfg, model, results, answer)

        g = sns.FacetGrid(data=probs, hue='variable', col='n_frags_1', row='n_frags_2')
        g.map(sns.kdeplot, 'value')
        g.set_titles("n_frags={col_name} vs {row_name}")
        g.add_legend()
        plt.show()

        if input("Plot coverage? (y/[n]) ").lower() == 'y':
            cmd = 'python {}/util/plot/show_coverage.py --dataset {} --contigs "{}"'.format(
                PARENT_DIR, dataset, ' '.join([ctg1, ctg2])
            )
            os.system(cmd)

if __name__ == '__main__':
    main()
