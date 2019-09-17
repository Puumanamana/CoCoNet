"""
Extract metrics for either one experiment or all the simulations
Plot a line chart with the extracted results
"""

from pathlib import Path
import os
import sys
from itertools import chain
from glob import glob

import pandas as pd
import sklearn.metrics

import seaborn as sns
import matplotlib.pyplot as plt

PARENT_DIR = os.path.join(sys.path[0], '..')
sys.path.insert(1, PARENT_DIR)

from experiment import Experiment

METRICS = {'nn': ['roc_auc_score', 'accuracy_score', 'cohen_kappa_score', 'f1_score'],
           'clustering': ['adjusted_rand_score', 'adjusted_mutual_info_score', 'homogeneity_score', 'completeness_score']}

def get_metrics_run(folder, single=False, mode='nn'):
    '''
    Get metrics for given run.
    Choice between NN metrics or clustering metrics
    '''
    config = Experiment(folder.stem, root_dir=PARENT_DIR)

    if mode == 'nn':
        filename = config.outputs['net']['test']
        pred_field = 'combined'
    else:
        filename = config.outputs['clustering']['assignments'].replace(',louvain', '')
        pred_field = 'clusters'

    if not os.path.exists(filename):
        print('{} not found. Ignoring file.'.format(filename))
        return [[]]

    test_results = pd.read_csv(filename)

    y_pred = test_results[pred_field].values
    y_pred_bin = (y_pred > 0.5).astype(int)
    y_test = test_results.truth.values

    if not single:
        coverage, n_samples = folder.stem.split('_')
    else:
        coverage, n_samples = (-1, -1)

    result = [[float(coverage), int(n_samples),
               metric, getattr(sklearn.metrics, metric)(y_test, y_pred)]
              if metric in ['roc_auc_score']+METRICS['clustering'] else
              [float(coverage), int(n_samples),
               metric, getattr(sklearn.metrics, metric)(y_test, y_pred_bin)]
              for metric in METRICS[mode]]

    return result

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--single', default=False, action="store_true")
    parser.add_argument('--path', type=str, default='.')
    parser.add_argument('--mode', type=str, default='nn', choices=['nn', 'clustering'])

    args = parser.parse_args()

    if args.single:
        config_paths = [Path(args.path)]
    else:
        config_paths = [Path(f) for f in glob('{}/output_data/*'.format(PARENT_DIR))
                        if Path(f).stem.replace('_', '').isdigit()]

    data = list(chain(*[get_metrics_run(path, single=args.single, mode=args.mode)
                        for path in config_paths]))
    data = (
        pd.DataFrame(data, columns=['coverage', 'nunber_of_samples', 'metric', 'score'])
        .dropna()
        .astype({'coverage': float,
                 'nunber_of_samples': int,
                 'metric': str,
                 'score': float})
        )

    if not args.single:
        (
            sns.catplot(x='nunber_of_samples', y='score', hue='coverage', col='metric',
                        kind='point', col_wrap=2, sharey=False, legend=False,
                        linestyles=["--"]*len(data.coverage.unique()),
                        markers=['^']*len(data.coverage.unique()),
                        data=data)
            .set_xlabels(fontsize=16)
            .set_ylabels(fontsize=16)
            .set_titles(fontweight='bold', fontsize=16)
            .add_legend(title='Coverage')
        )
        plt.savefig("{}_metrics.png".format(args.mode), dpi=300)
        plt.show()

    else:
        print(data)
        import ipdb;ipdb.set_trace()
