import os
from os.path import basename

import sys
from glob import glob
import pandas as pd
import torch

import sklearn.metrics

import seaborn as sns
import matplotlib.pyplot as plt

PARENT_DIR = os.path.join(sys.path[0], '..')
sys.path.insert(1, PARENT_DIR)

from experiment import Experiment
from nn_training import initialize_model,load_data,get_labels

metrics = ['roc_auc_score', 'accuracy_score', 'precision_score', 'recall_score', 'cohen_kappa_score', 'f1_score']

def load_network(config):
    config.set_input_shapes()

    model = initialize_model(config.model_type, config)

    checkpoint = torch.load(config.outputs['model'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model

def get_test(config):
    generators = load_data(config, mode='test')
    x_test = list(map(next, generators))

    y_test = get_labels(config.outputs['fragments']['test'])

    return x_test, y_test

sim_folders = [f for f in glob('{}/output_data/*'.format(PARENT_DIR))
               if basename(f).replace('_', '').isdigit()]
data = []

for f in sim_folders:
    config = Experiment(basename(f), root_dir=PARENT_DIR)

    coverage, n_samples = os.path.basename(f).split('_')    

    if not os.path.exists(config.outputs['model']):
        continue

    model = load_network(config)
    X_test, y_test = get_test(config)

    y_pred = model(*X_test)['combined'].detach().numpy()[:, 0]
    y_pred_bin = (y_pred > 0.5).astype(int)

    data += [[coverage, n_samples, metric, getattr(sklearn.metrics, metric)(y_test, y_pred)]
             if metric == 'roc_auc_score' else
             [coverage, n_samples, metric, getattr(sklearn.metrics, metric)(y_test, y_pred_bin)]
             for metric in metrics]

data = pd.DataFrame(data, columns=['coverage_per_sample', 'nb_samples', 'metric', 'score'])

(
    sns.catplot(x='nb_samples', y='score', hue='coverage_per_sample', col='metric',
                kind='point', col_wrap=3, sharey=False,
                linestyles=["--"]*len(data.coverage_per_sample.unique()),
                markers=['^']*len(data.coverage_per_sample.unique()),
                data=data)
    .set_xticklabels(rotation=0)
    .set_xlabels(fontsize=16)
    .set_ylabels(fontsize=16)
    .set_titles(fontweight='bold', fontsize=16)
)

plt.show()
