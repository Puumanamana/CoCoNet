"""
Extract metrics for either one experiment or all the simulations
Plot a line chart with the extracted results
"""

from pathlib import Path
import os
import sys
from itertools import chain
from glob import iglob
import argparse
import re

import pandas as pd
import numpy as np
import sklearn.metrics
from Bio import SeqIO

import seaborn as sns
import matplotlib.pyplot as plt

# sns.set(rc={'figure.figsize':(15, 10)})
sns.set_style("darkgrid")

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
                          # 'adjusted_mutual_info_score',
                          'fowlkes_mallows_score',
                          'homogeneity_score', 'completeness_score']}

CATPLOT_PARAMS = {
    'clustering':{
        'x': 'metric', 'y': 'score', 'hue': 'method', 'units': 'iter',
        'col': 'coverage', 'row': 'samples', 'sharey': False,
        'height': 3, 'aspect': 2,
        'facet_kws': {'ylim': [-0.1, 1.1]},
        'margin_titles': True,
        'kind': 'box', 'showfliers': False,
        # 'kind': 'strip', 's': 6, 'linewidth': 1, 'alpha': 0.7
    },
    'nn': {
        'x': 'samples', 'y': 'score', 'hue': 'coverage', 'units': 'iter',
        'col': 'number of genomes', 'row': 'metric', 'sharey': 'row',
        'height': 3, 'aspect': 2,
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
        # Params = (n_samples, coverage, n_genomes, n_iter)
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
        return (y_test, y_pred, params, True)

    data = data[data.contigs.str.contains('|', regex=False)]
    y_pred = data.clusters
    y_test = pd.factorize(data.contigs.str.split('|').str[0])[0]
    valid = check_experiment(y_test)
    
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
        elif method == 'metabat2':
            assignments = load_metabat2(out_path)
        else:
            print('Unknown method: {}'.format(method))
            import ipdb;ipdb.set_trace()

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
    
    assignments = pd.read_csv(filename, header=None,
                              names=['contigs', 'clusters'])
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
        print('{} not found. Ignoring file.'.format(filename), end='\r')
        return [[]]

    test_results = pd.read_csv(filename)

    if mode == 'clustering':
        y_test, y_pred, params, valid = get_outputs(path, test_results)
        if not valid:
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

def collect_all_metrics(args):
    if args.path:
        config_paths = [path for path in args.path if path]
    else:
        if args.nvir > 0:
            config_paths = [Path(f) for f in iglob('{}/{}/*'.format(PARENT_DIR, DIRS['CoCoNet']))
                            if re.match(r'^{}_\d+_\d+_\d+$'.format(args.nvir), Path(f).stem)]
        else:
            config_paths = [Path(f) for f in iglob('{}/{}/*'.format(PARENT_DIR, DIRS['CoCoNet']))
                            if Path(f).stem.replace('_', '').isdigit()]

    data = chain(*[get_metrics_run(path,
                                   mode=args.mode,
                                   feature_name=args.feature)
                   for path in config_paths])

    data = [x for x in data if len(x) > 0]

    if args.mode == 'clustering':
        for method in ['CONCOCT', 'metabat2']:
            data_method = load_external_results(config_paths, method)
            data += data_method

    columns = pd.Series({'method': str, 'metric': str, 'score': float,
                         'iter': int, 'samples': int, 'coverage': int, 'number of genomes': int})
    columns = columns[:len(data[0])]

    if args.path:
        columns = columns.rename({'iter': 'dataset'})
        columns.loc['dataset'] = str

    data = pd.DataFrame(data, columns=columns.index).astype(columns).dropna()

    return data


def plot_metrics(data, nvir, mode, fs=12, nw=False, output=None):

    data['metric'] = (data['metric']
                      .str.replace('_score', '')
                      .str.capitalize()
                      .replace({'Adjusted_rand': 'ARI',
                                'Adjusted_mutual_info': 'AMI',
                                'Fowlkes_mallows': 'FMI',
                                'Roc_auc': 'AUC'}))

    is_sim = 'coverage' in data.columns

    if is_sim and mode == 'clustering':
        data = data[data['number of genomes'] == nvir]
        data.metric = data.metric.str.replace('_', '\n')
    if not is_sim:
        CATPLOT_PARAMS[mode]['kind'] = 'bar'
        if mode == 'clustering':
            CATPLOT_PARAMS[mode]['row'] = 'dataset'
            del CATPLOT_PARAMS[mode]['col']
            del CATPLOT_PARAMS[mode]['units']
            del CATPLOT_PARAMS[mode]['showfliers']
        else:
            CATPLOT_PARAMS[mode]['x'] = 'metric'
            CATPLOT_PARAMS[mode]['hue'] = 'dataset'
            del CATPLOT_PARAMS[mode]['col']
            del CATPLOT_PARAMS[mode]['row']
            del CATPLOT_PARAMS[mode]['linestyle']

    handle = sns.catplot(legend=False, data=data,
                         **CATPLOT_PARAMS[mode])
    handle.set(ylabel='')

    if is_sim:
        if mode == 'clustering':
            (handle.set_xlabels(fontsize=fs)
             .set_titles('coverage: {col_name}X', size=fs+2, fontweight='bold')
             # .set_titles('{row_name} samples (cov={col_name}X)', size=fs+2, fontweight='bold')
             # .set_titles('')
             .set(xlabel='', ylabel='')
             .add_legend()
            )
        else:
            handle.add_legend(title='Coverage')
            handle.set_titles('# genomes: {col_name}', size=fs+2, fontweight='bold')
            # handle.set(ylabel='{row_name}')

            rownames = data[CATPLOT_PARAMS[mode]['row']].unique()
            colnames = data[CATPLOT_PARAMS[mode]['col']].unique()

            for i, ax_i in enumerate(handle.axes.flat):
                row, col = i // len(colnames), i % len(colnames)

                ax_i.set_ylim([ax_i.get_ylim()[0], 1])

                if col == 0:
                    ax_i.set_ylabel("{:}".format(rownames[row]), fontsize=fs, fontweight='bold')
                if row != 0:
                    ax_i.set_title('')
                if row == len(rownames)-1:
                    ax_i.set_xticklabels(ax_i.get_xticklabels(), fontsize=fs)
                    ax_i.set_xlabel(ax_i.get_xlabel().capitalize(), fontsize=fs, fontweight='bold')

    else:
        handle.add_legend()

    for (i, j, _), fd in handle.facet_data():
        ax = handle.facet_axis(i, j)
        ax.tick_params(labelbottom=True)
        if fd.empty:
            ax.set_axis_off()
            ax.set_title('')

    plt.subplots_adjust(right=0.9, hspace=0.3)
    plt.savefig("{}.png".format(output), dpi=300)

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
        if args.mode == 'clustering':
            suffix = '{}_{}'.format(suffix, args.nvir)

    output_radical = '{}/figures/{}_metrics-{}'.format(PARENT_DIR, args.mode, suffix)

    plot_metrics(data, args.nvir, args.mode, nw=args.nw, output=output_radical)

    if not args.path and args.mode == 'clustering':
        data.drop('number of genomes', axis=1, inplace=True)
        (data
         .set_index(["metric", "method", "coverage", "samples"])
         .sort_index()
         .to_csv("{}.csv".format(output_radical)))
        (data.groupby(["metric", "method"]).score.agg(['min', 'max', 'mean', 'median'])
         .to_csv("{}_summary.csv".format(output_radical)))
        
    else:
        data.to_csv("{}.csv".format(output_radical))

if __name__ == '__main__':
    main()
