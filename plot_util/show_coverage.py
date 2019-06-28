import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import h5py

inputdir = os.path.expanduser("~/CoCoNet/input_data")

def plot_coverage(dataset, *viruses):
    cov_h5 = h5py.File("{}/{}/coverage_virus.h5".format(inputdir,dataset))

    dfs = []

    for virus in viruses:
        df = pd.DataFrame(cov_h5.get(virus)[:]).T.reset_index()
        df.columns = ["position"]+["sample_{}".format(i) for i in range(df.shape[1]-1)]  
        df["V_id"] = virus
        dfs.append(df)

    dfs = pd.concat(dfs).melt(id_vars=["V_id","position"])
    dfs.columns = ["V_id","position","sample","coverage"]

    g = sns.FacetGrid(data=dfs, col="sample", hue="V_id", col_wrap=3, sharey=False)
    g.map(plt.plot,"position","coverage")
    plt.show()

if __name__ == '__main__':
    import sys
    
    plot_coverage("sim", *sys.argv[1:])

    
    
