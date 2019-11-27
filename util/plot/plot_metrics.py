"""
Extract metrics for either one experiment or all the simulations
Plot a line chart with the extracted results
"""
from time import time
from pathlib import Path
import os
import sys
from itertools import chain
from glob import iglob
import argparse


import pandas as pd
import numpy as np
import sklearn.metrics
from Bio import SeqIO

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(**{'context': 'paper', 'style': 'darkgrid'})
fs = 10

rc = {'axes.labelsize': fs+2,
      'legend.fontsize': fs,
      'axes.titlesize': fs+2,
      'xtick.labelsize': fs-2, 'ytick.labelsize': fs-2}

plt.rcParams.update(**rc)

PARENT_DIR = os.path.join(sys.path[0], '../..')
sys.path.insert(1, PARENT_DIR)

from experiment import Experiment

DIRS = {
    'input': 'input_data',
    'CoCoNet': 'output_data',
    'others': '{}/../concurrence_results'.format(PARENT_DIR)
}

METRICS = {'nn': ['roc_auc_score', 'accuracy_score', 'f1_score'],
           'clustering': ['adjusted_rand_score',
                          'homogeneity_score',
                          'completeness_score']}

CATPLOT_PARAMS = {
    'clustering': {
        'x': 'metric', 'y': 'score', 'hue': 'method', 'units': 'iter',
        'row': 'settings', 'sharey': False, 'col': 'bins',
        'row_order': ['{} samples ({}X)'.format(s, c) for s in [4, 15] for c in [3, 10]],
        'height': 3, 'aspect': 2,
        'facet_kws': {'ylim': [-0.1, 1.1]},
        'margin_titles': True,
        'kind': 'bar', #'showfliers': False,
        # 'kind': 'strip', 's': 6, 'linewidth': 1, 'alpha': 0.7
    },
    'nn': {
        'x': 'samples', 'y': 'score', 'hue': 'coverage', 'units': 'iter',
        'hue_order': ['3X', '10X'], 'margin_titles': True,
        'col': 'bins', 'row': 'metric', 'sharey': 'row',
        'height': 2, 'aspect': 1.5,
        'kind': 'box', 'showfliers': False,
        # 'kind': 'strip', 's': 7, 'linewidth': 1, 'alpha': 0.7
    }
}

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, nargs='+', default=[], help="List of path (space separated)")
    parser.add_argument('--nvir', type=int, default=1000)
    parser.add_argument('--mode', type=str, default='clustering', choices=['nn', 'clustering'])
    parser.add_argument('--feature', type=str, default='combined', choices=['composition', 'coverage', 'combined'])
    parser.add_argument('--nw', action='store_true', default=False)

    args = parser.parse_args()
    args.path = [Path(path) for path in args.path]

    if args.mode == 'nn':
        args.nvir = 0

    return args

def get_outputs(path, data, params_only=False):
    if path.stem.replace('_', '').isdigit():
        params = path.stem.split('_')[::-1]
    else:
        params = [path.stem]

    if params_only:
        return params

    if 'truth.csv' in os.listdir(path):
        y_test = pd.read_csv('{}/truth.csv'.format(path), header=None,
                             names=['contigs', 'clusters'])
        in_common = np.intersect1d(y_test.contigs, data.contigs)
        y_pred = data.set_index('contigs').loc[in_common, 'clusters']
        y_test = y_test.set_index('contigs').loc[in_common, 'clusters']
        valid = True
    else:
        data = data[data.contigs.str.contains('|', regex=False)]
        y_pred = data.clusters
        y_test = pd.factorize(data.contigs.str.split('|').str[0])[0]
        valid = check_experiment(y_test)

    count_complete_bins(y_test, y_pred)

    return (y_test, y_pred, params, valid)

def check_experiment(truth):
    counts = np.unique(truth, return_counts=True)[1]
    return sum(counts > 1) >= 10

def load_external_results(input_paths, method):
    output_paths = [Path("{}/{}/{}".format(DIRS['others'], path.stem, method))
                    for path in input_paths]

    results = []

    for in_path, out_path in zip(input_paths, output_paths):
        if method == 'CONCOCT':
            assignments = load_concoct(out_path)
        elif method == 'Metabat2':
            assignments = load_metabat2(out_path)
        else:
            print('Unknown method: {}'.format(method))

        if assignments.size == 0:
            continue

        y_test, y_pred, params, valid = get_outputs(in_path, assignments)

        if valid:
            results += [[method, metric, getattr(sklearn.metrics, metric)(y_test, y_pred, average_method='arithmetic')] + params
                        if metric == 'adjusted_mutual_info_score' else
                        [method, metric, getattr(sklearn.metrics, metric)(y_test, y_pred)] + params
                        for metric in METRICS['clustering']]
    return results

def load_concoct(out_path):
    filename = "{}/concoct_clustering_gt1000.csv".format(out_path)

    if not os.path.exists(filename):
        return pd.DataFrame([])
    
    assignments = pd.read_csv(filename)
    assignments.columns = ['contigs', 'clusters']

    return assignments

def load_metabat2(out_path):
    assignments = []

    for i, fasta in enumerate(iglob('{}/*.fa'.format(out_path))):
        for seq in SeqIO.parse(fasta, 'fasta'):
            assignments.append([seq.id, i])

    all_contigs = pd.read_csv(f'{out_path}/../metabat_coverage_table.tsv',
                              usecols=['contig_id'], sep='\t').contig_id.values

    if not assignments:
        assignments = np.vstack([all_contigs, np.arange(len(all_contigs))]).T
    else:
        assignments = np.array(assignments)
    not_binned = np.setdiff1d(all_contigs, assignments[:, 0])

    not_binned = np.vstack([not_binned, np.arange(len(not_binned)) + assignments[:, 1].astype(int).max()+1]).T
    assignments = pd.DataFrame(np.vstack([assignments, not_binned]),
                               columns=['contigs', 'clusters'])
    assignments.clusters = assignments.clusters.astype(int)
    return assignments

def get_metrics_run(path, mode='nn', feature_name=None):
    '''
    Get metrics for given run.
    Choice between NN metrics or clustering metrics
    '''
    
    config = Experiment(path.stem, root_dir=PARENT_DIR,
                        in_dir=DIRS['input'],
                        out_dir=DIRS['CoCoNet'])

    if mode == 'nn':
        filename = config.outputs['net']['test']
    else:
        filename = config.outputs['clustering']['refined_assignments']
        feature_name = 'clusters'

    if not os.path.exists(filename):
        print('{} not found. Ignoring file.'.format(filename))
        return [[]]

    print('\033[1mProcessing {}.\033[0m'.format(path.stem))

    test_results = pd.read_csv(filename)

    if mode == 'clustering':
        to_exclude = pd.read_csv(config.singleton_ctg_file, sep='\t')

        if len(to_exclude) > 0:
            test_results = test_results.set_index('contigs').drop(to_exclude.contigs).reset_index()

    if mode == 'clustering':
        test_results.columns = ['contigs', 'clusters', 'truth']
        y_test, y_pred, params, valid = get_outputs(path, test_results)

        if not valid:
            print('Not enough bins to work with. Skipping dataset')
            return [[]]

    else:
        y_test = test_results['truth'].values
        y_pred = test_results[feature_name].values
        params = get_outputs(path, test_results, params_only=True)

    y_pred_bin = (y_pred > 0.5).astype(int)

    metric_res = []
    for metric in METRICS[mode]:
        kwargs = {}
        if metric == 'adjusted_mutual_info_score':
            kwargs['average_method'] = 'arithmetic'
        if metric in ['roc_auc_score']+METRICS['clustering']:
            score = getattr(sklearn.metrics, metric)(y_test, y_pred, **kwargs)
        else:
            score = getattr(sklearn.metrics, metric)(y_test, y_pred_bin, **kwargs)
        metric_res.append(['CoCoNet', metric, score] + params)
    return metric_res


def count_complete_bins(y_test, y_pred):

    def get_imax(x):
        indices, freqs = np.unique(x, return_counts=True)
        imax = indices[freqs.argmax()]
        return ','.join(map(str, [len(x), imax, freqs.max()]))

    combined = pd.DataFrame({'pred': y_pred, 'truth': y_test}).dropna(how='any')
    combined['truth'] = 'V' + combined.truth.astype(str)
    bin_sizes = combined.groupby('truth').pred.agg(len)

    assessment = combined.groupby('pred').truth.agg(get_imax).str.split(',', expand=True)
    assessment.columns = ['csize', 'bin_imax', 'bin_max']
    assessment.bin_max = assessment.bin_max.astype(int)
    assessment.csize = assessment.csize.astype(int)

    assessment['homogeneity'] = assessment.bin_max / assessment.csize
    assessment['completeness'] = assessment.bin_max / bin_sizes.loc[assessment['bin_imax']].values

    cond_homogeneous = assessment.homogeneity == 1
    cond_complete = assessment.completeness == 1
    cond_ns = assessment.csize > 1

    info = [
        'all ({} predicted, {} true): {} homogeneous, {} complete, {} exact'.format(
            len(assessment),
            len(bin_sizes),
            sum(cond_homogeneous),
            sum(cond_complete),
            sum(cond_homogeneous & cond_complete)
        ),
        'non singletons ({} predicted, {} true): {} homogeneous, {} complete, {} exact'.format(
            sum(cond_ns),
            sum(bin_sizes > 1),
            sum(cond_homogeneous & cond_ns),
            sum(cond_complete & cond_ns),
            sum(cond_homogeneous & cond_complete & cond_ns)
        )
    ]
    print('\n'.join(info))

def collect_all_metrics(args):
    if args.path:
        config_paths = [path for path in args.path if path]
    else:
        # if args.nvir > 0:
        #     config_paths = [Path(f) for f in iglob('{}/{}/*'.format(PARENT_DIR, DIRS['CoCoNet']))
        #                     if re.match(r'^{}_\d+_\d+_\d+$'.format(args.nvir), Path(f).stem)]
        # else:
        config_paths = [Path(f) for f in iglob('{}/{}/*'.format(PARENT_DIR, DIRS['CoCoNet']))
                        if Path(f).stem.replace('_', '').isdigit()]

    data = chain(*[get_metrics_run(path,
                                   mode=args.mode,
                                   feature_name=args.feature)
                   for path in config_paths])

    data = [x for x in data if len(x) > 0]

    if args.mode == 'clustering':
        for method in ['CONCOCT', 'Metabat2']:
            print('\033[1mProcessing {}\033[0m'.format(method))
            data_method = load_external_results(config_paths, method)
            data += data_method

    columns = pd.Series({'method': str, 'metric': str, 'score': float,
                         'iter': int, 'samples': int, 'coverage': int, 'bins': int})

    columns = columns[:len(data[0])]

    if args.path:
        columns = columns.rename({'iter': 'dataset'})
        columns.loc['dataset'] = str

    data = pd.DataFrame(data, columns=columns.index).astype(columns).dropna()

    if not args.path:
        data.coverage = data.coverage.astype(str) + 'X'
        data['settings'] = ['{} samples ({})'.format(sample, cov)
                            for sample, cov in data[['samples', 'coverage']].values]

    return data

def plot_metrics(data, mode, nw=False, output=None):

    data['metric'] = (data['metric']
                      .str.replace('_score', '')
                      .str.capitalize()
                      .replace({'Adjusted_rand': 'ARI',
                                'Adjusted_mutual_info': 'AMI',
                                'Fowlkes_mallows': 'FMI',
                                'Roc_auc': 'AUC'}))

    is_sim = 'bins' in data.columns

    if is_sim and mode == 'clustering':
        # data = data[data['number of bins'] == nvir]
        data.metric = data.metric.str.replace('_', '\n')
    if not is_sim:
        CATPLOT_PARAMS[mode]['kind'] = 'bar'
        if mode == 'clustering':
            CATPLOT_PARAMS[mode]['col'] = 'dataset'
            del CATPLOT_PARAMS[mode]['row']
            del CATPLOT_PARAMS[mode]['units']
        else:
            CATPLOT_PARAMS[mode]['x'] = 'metric'
            CATPLOT_PARAMS[mode]['hue'] = 'dataset'
            del CATPLOT_PARAMS[mode]['hue_order']
            del CATPLOT_PARAMS[mode]['col']
            del CATPLOT_PARAMS[mode]['row']
            del CATPLOT_PARAMS[mode]['linestyle']

    g = sns.catplot(legend=False, data=data, **CATPLOT_PARAMS[mode])

    for ax in g.axes.flat:
        plt.setp(ax.texts, text="")

    g.add_legend().despine(left=True)

    if is_sim:
        if mode == 'clustering':
            (g.set_titles(col_template='#bins: {col_name}', row_template='{row_name}')
             .set(xlabel='', ylabel=''))
        else:
            (g.set_titles(col_template='#bins: {col_name}', row_template='')
             .set_ylabels(''))
            rownames = data[CATPLOT_PARAMS[mode]['row']].unique()

        plt.subplots_adjust(right=0.85, hspace=0.3)
    else:
        g.set(xlabel='', ylabel='').set_titles(col_template='{col_name}')
        plt.subplots_adjust(right=0.8, hspace=0.3)

    for (row, col, _), facet_data in g.facet_data():
        ax_k = g.facet_axis(row, col)

        if mode == 'clustering':
            ax_k.tick_params(labelbottom=True)
            ax_k.set_ylim([ax_k.get_ylim()[0], 1.1])

            if col > 0:
                ax_k.set_yticklabels([])

        if is_sim and mode == 'nn':
            ax_k.set_ylim([ax_k.get_ylim()[0], 1])

            if col == 0:
                ax_k.set_ylabel(rownames[row])

        if facet_data.empty:
            ax_k.set_axis_off()
            ax_k.set_title('')

    plt.savefig("{}.pdf".format(output), transparent=True)

    if not nw:
        plt.show()

def main():
    args = parse_args()
    print(args)

    data = collect_all_metrics(args)

    if args.path:
        suffix = '_'.join([path.stem for path in args.path])
    else:
        suffix = 'sim'

    fig_nb = [('nn', True), ('clustering', True), ('clustering', False), ('nn', False)].index((args.mode, 'sim' in suffix))

    output_radical = '{}/figures/Figure {}-{}_metrics-{}'.format(PARENT_DIR, fig_nb+3, args.mode, suffix)

    plot_metrics(data, args.mode, nw=args.nw, output=output_radical)

    if not args.path and args.mode == 'clustering':
        # data.drop('number of genomes', axis=1, inplace=True)
        (data
         .set_index(["metric", "method", "bins", "coverage", "samples"])
         .sort_index()
         .to_csv("{}.csv".format(output_radical)))
        (data.groupby(["metric", "method"]).score.agg(['min', 'max', 'mean', 'median'])
         .to_csv("{}_summary.csv".format(output_radical)))

    else:
        data.to_csv("{}.csv".format(output_radical))

if __name__ == '__main__':
    main()
