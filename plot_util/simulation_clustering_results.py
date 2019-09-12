import os, sys
from glob import glob
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from plot_clustering_outcome import plot_scores

PARENT_DIR = os.path.join(sys.path[0], '..')

sim_folders = [f for f in glob('{}/output_data/*'.format(PARENT_DIR))
               if os.path.basename(f).replace('_', '').isdigit()]

data = []
for f in sim_folders:
    print(f)
    coverage, n_samples = os.path.basename(f).split('_')

    try:
        assignments = pd.read_csv("{}/leiden_nf30.csv".format(f), index_col=0)
    except FileNotFoundError:
        print("Not found", f)
        continue

    pred = assignments.clusters.values
    truth = assignments.truth.values

    metrics = plot_scores(pred, truth, plot=False)
    metrics['coverage_per_sample'] = int(coverage)
    metrics['nb_samples'] = int(n_samples)

    data.append(metrics)

data = pd.concat(data)

print(data.head())
print(data.coverage_per_sample.value_counts())
print(data.nb_samples.value_counts())

g = sns.catplot(x='nb_samples', y='score', hue='coverage_per_sample', col='metric', kind='point',
                sharey=False,
                linestyles=["--"]*len(data.coverage_per_sample.unique()),
                markers=['^']*len(data.coverage_per_sample.unique()),
                data=data)
(
    g.set_xticklabels(rotation=0)
    .set_xlabels(fontsize=16)
    .set_ylabels(fontsize=16)
    .set_titles(fontweight='bold', fontsize=16)
)
plt.show()
