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

import pandas as pd
import numpy as np
import sklearn.metrics

import seaborn as sns
import matplotlib.pyplot as plt

PARENT_DIR = os.path.join(sys.path[0], '../..')
sys.path.insert(1, PARENT_DIR)

from experiment import Experiment

DIRS = {
    'input': 'input_data',
    'CoCoNet': 'output_data',
    'CONCOCT': '{}/../CONCOCT/CONCOCT_results'.format(PARENT_DIR),
}

METRICS = {'nn': ['roc_auc_score', 'accuracy_score', 'cohen_kappa_score', 'f1_score'],
           'clustering': ['adjusted_rand_score', 'homogeneity_score', 'completeness_score']}

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='', help="List of path (semicolon separated)")
    parser.add_argument('--nvir', type=int, default=1000)
    parser.add_argument('--mode', type=str, default='nn', choices=['nn', 'clustering'])
    parser.add_argument('--feature', type=str, default='combined', choices=['composition', 'coverage', 'combined'])

    args = parser.parse_args()

    return args

def get_outputs(path, data, params_only=False):
    if path.stem.replace('_', '').isdigit():
        # Params = (n_samples, coverage, n_genomes)
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
        return (y_test, y_pred, params)

    data = data[data.contigs.str.contains('|', regex=False)]
    y_pred = data.clusters
    y_test = pd.factorize(data.contigs.str.split('|').str[0])[0]
    return (y_test, y_pred, params)

def load_concoct_results(input_paths):
    output_paths = [Path("{}/{}".format(DIRS['CONCOCT'], path.stem)) for path in input_paths]

    results = []

    for in_path, out_path in zip(input_paths, output_paths):
        assignments = pd.read_csv("{}/concoct_clustering_gt1000.csv".format(out_path), header=None,
                                  names=['contigs', 'clusters'])
        y_test, y_pred, params = get_outputs(in_path, assignments)

        results += [['CONCOCT', metric, getattr(sklearn.metrics, metric)(y_test, y_pred)] + params
                    for metric in METRICS['clustering']]
    return results

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
        y_test, y_pred, params = get_outputs(path, test_results)
    else:
        y_test = test_results['truth'].values
        y_pred = test_results[feature_name].values
        params = get_outputs(path, test_results, params_only=True)

    y_pred_bin = (y_pred > 0.5).astype(int)

    metric_res = []
    for metric in METRICS[mode]:
        if metric in ['roc_auc_score']+METRICS['clustering']:
            score = getattr(sklearn.metrics, metric)(y_test, y_pred)
        else:
            score = getattr(sklearn.metrics, metric)(y_test, y_pred_bin)
        metric_res.append(['CoCoNet', metric, score] + params)
    return metric_res

def collect_all_metrics(args):
    if args.path != '':
        config_paths = [Path(path) for path in args.path.split(' ')]
    else:
        config_paths = [Path(f) for f in iglob("{}/{}/*".format(PARENT_DIR, DIRS['CoCoNet']))
                        if Path(f).stem.startswith("{}_".format(args.nvir))]

    data = chain(*[get_metrics_run(path,
                                   mode=args.mode,
                                   feature_name=args.feature)
                   for path in config_paths])

    data = [x for x in data if len(x) > 0]

    if args.mode == 'clustering':
        data_concoct = load_concoct_results(config_paths)
        data += data_concoct

    columns = pd.Series({'method': str, 'metric': str, 'score': float,
                         'number of samples': int, 'coverage': int, 'number of genomes': int})
    columns = columns[:len(data[0])]

    if args.path != '':
        columns = columns.rename({'number of samples': 'dataset'})
        columns.loc['dataset'] = str

    data = pd.DataFrame(data, columns=columns.index).astype(columns).dropna()

    return data

def plot_metrics(data, nvir, mode, fs=12):
    catplot_params = {
        'x': 'number of samples',
        'y': 'score',
        'hue': 'method',
        'col': 'coverage',
        'row': 'metric',
        'kind': 'bar',
        'sharey': False
    }

    is_sim = 'coverage' in data.columns

    if is_sim:
        data = data[data['number of genomes'] == nvir]

        if mode == 'nn':
            catplot_params['sharey'] = 'row'
            catplot_params['kind'] = 'point'
            del catplot_params['hue']
            catplot_params.update({
                'linestyles': ["--"]*len(data.coverage.unique()),
                'markers': ['^']*len(data.coverage.unique())
            })

    else:
        del catplot_params['col']
        catplot_params['hue'] = 'method'
        catplot_params['x'] = 'dataset'

    handle = sns.catplot(legend=False, data=data, height=3, aspect=1,
                         **catplot_params)
    axes = handle.axes.flat

    if is_sim:
        (handle.set_xlabels(fontsize=fs)
         .set_titles('{col_var} = {col_name}X', size=fs)
        )

        n_cov = len(data.coverage.unique())
        metrics = [x.replace('_', ' ').capitalize() for x in data.metric.unique()]

        for i, ax in enumerate(axes):
            if i//n_cov != 0:
                ax.set_title('')
            if i%n_cov == 0:
                ax.set_ylabel(metrics[i//n_cov], fontsize=fs)

        axes[7].legend(bbox_to_anchor=(1.05, 0.5), loc=2, fontsize=fs)

    else:
        handle.add_legend()
    plt.tight_layout()
    # plt.savefig("{}/figures/{}_metrics.png".format(PARENT_DIR, mode), dpi=300)
    plt.show()


def main():
    args = parse_args()

    data = collect_all_metrics(args)

    plot_metrics(data, args.nvir, args.mode)

    return data

if __name__ == '__main__':
    result = main()

    import ipdb;ipdb.set_trace()
