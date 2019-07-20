import sys
import pandas as pd
import numpy as np
from Bio import SeqIO
from progressbar import progressbar
from itertools import combinations

sys.path.append("..")
from util import get_kmer_frequency

import seaborn as sns
import matplotlib.pyplot as plt

def display_distr(df):
    g = sns.FacetGrid(data=df,hue='label',col='kmer',sharex=False,sharey=False)
    g.map(sns.distplot,'dist').add_legend()
    plt.show()

assembly = "/home/cedric/projects/viral_binning/CoCoNet/input_data/sim/assembly_gt2048.fasta"
root_dir = "/home/cedric/projects/viral_binning/CoCoNet/output_data/sim"

pairs = pd.read_csv("{}/pairs_test.csv".format(root_dir),header=[0,1],index_col=0,nrows=1000)
genomes = { seq.id: str(seq.seq) for seq in SeqIO.parse(assembly,"fasta") }
# species = set(pairs.A.sp.tolist() + pairs.B.sp.tolist())

kmer_max = 6
rc=False

y = (pairs.A.sp.str.split("_").str[0] == pairs.B.sp.str.split("_").str[0]).astype(int)

distance_list = []

limits = np.cumsum([0] + [ int(4**k/(1+rc)) for k in range(3,kmer_max+1) ])

X_combined = np.zeros([2,len(y),limits[-1]])

for kmer in range(3,kmer_max+1):
    X = np.zeros([2,len(y),int(4**kmer/(1+rc))])

    for i,(spA,startA,endA,spB,startB,endB) in progressbar(enumerate(pairs.values),max_value=len(y)):
        X[0,i,:] = get_kmer_frequency(genomes[spA][startA:(endA)],kmer_list=[kmer],rc=rc)
        X[1,i,:] = get_kmer_frequency(genomes[spB][startB:(endB)],kmer_list=[kmer],rc=rc)

        X_combined[0,i,limits[kmer-3]:limits[kmer-2]] = X[0,i,:]
        X_combined[1,i,limits[kmer-3]:limits[kmer-2]] = X[1,i,:]

    distances = np.sqrt(np.sum((X[0]-X[1])**2,axis=1)) / (4**kmer)

    distance_list.append(pd.DataFrame({'dist': distances,
                                       'label': y,
                                       'kmer': kmer}))

dists = [ np.sqrt(np.sum(np.square(X_combined[0][:,limits[i]:limits[i+1]]-X_combined[1][:,limits[i]:limits[i+1]]),axis=1)) / limits[-1]
          for i in range(3) ]

distances = np.max(dists,axis=0)
# distances = np.sqrt(np.sum(np.square(X_combined[0]-X_combined[1]),axis=1)) / limits[-1]
distance_list.append(pd.DataFrame({'dist': distances,
                                   'label': y,
                                   'kmer': "all_kmer"}))
df_dists = pd.concat(distance_list)

display_distr(df_dists)

df = pd.concat([distance_list[1],distance_list[2]],axis=1).iloc[:,[0,1,3]].reset_index()
df.columns = ['ctg_idx','dist4','label','dist5']

sns.scatterplot(data=df,x="dist4",y="dist5",hue="label",edgecolor=None,alpha=0.2)
plt.show()

for i1,i2 in combinations(range(len(distance_list)),2):
    ax = sns.kdeplot(distance_list[i1].dist[distance_list[i1].label==0], distance_list[i2].dist[distance_list[i2].label==0],
                     cmap="Blues", shade=True, shade_lowest=False)
    ax = sns.kdeplot(distance_list[i1].dist[distance_list[i1].label==1], distance_list[i2].dist[distance_list[i2].label==1],
                     cmap="Reds", shade=True, shade_lowest=False)
    plt.show()
