import os

import numpy as np
import pandas as pd
import h5py

import sklearn.metrics
from umap import UMAP
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns


METRICS = ['adjusted_rand_score',
           # 'adjusted_mutual_info_score',
           'homogeneity_score',
           'completeness_score']

def nuniques(x):
    return len(set(x))

def plot_scores(truth,pred,metrics=METRICS,name="CoCoNet"):
    
    scores = pd.Series({
        metric: getattr(sklearn.metrics,metric)(truth,pred)
        for metric in metrics
    }, name="score")

    scores.index.name = "metric"

    fig,ax = plt.subplots()
    sns.barplot(x="metric",y="score",data=scores.reset_index(),ax=ax)

def plot_purity(assignments):
    # clusters_grouped = assignments.groupby('clusters')['truth'].agg([lambda x: len(set(x)),len])
    # clusters_grouped.columns = ["purity","csize"]

    clusters_grouped  = (assignments
                        .groupby('clusters')['truth']
                        .agg([lambda x: x.value_counts().max(),len,nuniques])
    )
    clusters_grouped.columns = ["purity","csize","nuniques"]

    clusters_grouped.purity = clusters_grouped.purity / clusters_grouped.csize

    clusters_grouped = clusters_grouped[clusters_grouped.csize>1]
    
    # clusters_purity = (clusters_grouped[clusters_grouped.csize>1].purity
    #                    .value_counts()
    #                    .sort_index())

    print(clusters_grouped)
    print("# clusters: {}".format(assignments.clusters.unique().shape[0]))

    fig,ax = plt.subplots(1,3)
    clusters_grouped.nuniques.value_counts().sort_index().plot(kind='bar',ax=ax[0])
    
    ax[1].scatter(clusters_grouped.purity+np.random.uniform(-.01,.01,clusters_grouped.shape[0]),
                  clusters_grouped.csize,
                  s=5,alpha=0.1)
    ax[1].set_xlabel("Cluster purity")
    ax[1].set_ylabel("Cluster size")
    sns.kdeplot(clusters_grouped.purity,ax=ax[2])

def get_components(h5,methods=["UMAP","PCA"]):

    handle = h5py.open(h5)
    data = np.vstack([ handle.get(k)[:] for k in handle.keys() ])

    for method in methods:
        output = "embeddings_{}.npy".format(method)
        
        if os.path.exists(output):
            continue
        
        if method.upper() == "UMAP":
            reducer = UMAP()
        elif method.upper() == "PCA":
            reducer = PCA(n_components=2)
        else:
            print("Unknown decomposition method. Aborting.")
            break

        embeddings = reducer.fit(data)
        np.save("embeddings_{}.npy".format(method),embeddings)

def plot_components(methods, labels):

    data = [ np.load("embeddings_{}.npy".format(method)) for method in methods ]

    n_frags = int(data[0].shape[1] / len(labels))
    
    truth = np.repeat([ int(l.split("_")[0][1:]) for l in labels], n_frags)

    
    fig, axes = plt.subplots(1,len(methods))
    
    for i,ax in enumerate(axes):
        ax.scatter(data[i][:,0],data[i][:,1],
                   s=5,c=truth)
        ax.set_xlabel("Embedding_1")
        ax.set_xlabel("Embedding_2")
        ax.set_title(methods[i])
    

if __name__ == '__main__':

    import sys
    dataset = sys.argv[1]

    root_dir = "/home/cedric/projects/viral_binning/CoCoNet/output_data/{}".format(dataset)
    assignments = pd.read_csv("{}/assignments_nf30.csv".format(root_dir),
                              index_col=0)
    
    if 'sim' in dataset:
        truth = [ int(x[1:]) for x in assignments.truth ]
    elif 'split' in dataset:
        mapping = pd.Series({ctg: i for i,ctg in enumerate(assignments.truth.unique())})
        truth = mapping.loc[assignments.truth]
    else:
        truth = pd.read_csv("{}/truth.csv".format(root_dir),header=None,index_col=0)[1].to_dict()
        assignments['truth'] = [ truth.get(ctg,None) for ctg in assignments.contigs ]
        assignments = assignments.dropna()
        assignments['truth'] = assignments['truth'].astype(int)

        truth = assignments['truth']
        
    pred = assignments.clusters.tolist()

    plot_purity(assignments)
    plot_scores(truth,pred)
    
    plt.show()

