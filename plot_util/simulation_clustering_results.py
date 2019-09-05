from os.path import basename
from glob import glob
import re
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from plot_clustering_outcome import plot_scores

sim_folders = glob('../output_data/vir_sim*')
data = []

for f in sim_folders:
    print(f)
    coverage = float(re.findall('cov-([\d\.]+)',basename(f))[0])
    n_samples = int(re.findall('(\d+)-samples',basename(f))[0])

    try:
        assignments = pd.read_csv("{}/assignments_nf30.csv".format(f), index_col=0)
    except FileNotFoundError:
        print("Not found",f)
        continue
    pred = assignments.clusters.values
    truth = assignments.truth.str.replace('V','').values.astype(int)

    metrics = plot_scores(pred,truth,plot=False)
    metrics['coverage_per_sample'] = coverage
    metrics['nb_samples'] = n_samples
    
    data.append(metrics)

data = pd.concat(data)

print(data.head())
print(data.total_coverage.value_counts())
print(data.n_samples.value_counts())

g = sns.catplot(x='nb_samples',y='score',hue='coverage_per_sample',col='metric',kind='point',
                linestyles=["--"]*len(data.coverage_per_sample.unique()),
                markers=['^']*len(data.coverage_per_sample.unique()),
                data=data)
(
    g.set_xticklabels(rotation=0)
    .set_xlabels(fontsize=16)
    .set_ylabels(fontsize=16)
    .set_titles(fontweight='bold',fontsize=16)
)
plt.show()
