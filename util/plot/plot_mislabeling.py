from scipy.special import comb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def count_max_mislabels(n_vir, n_ctg, max_ctg_per_virus):
    false_negatives = comb(max_ctg_per_virus, 2) * n_vir
    total_negatives = comb(n_ctg, 2)

    return false_negatives / total_negatives

if __name__ == '__main__':

    calc = []
    
    for nv in [500, 1000, 5000, 10000]:
        for nc_factor in [10, 20]:
            nc = nc_factor*nv
            for m in [20, 50, 100]:
                if m >= 10*nc/nv:
                    continue
                calc.append([nv, nc, m, count_max_mislabels(nv, nc, m)])

    # combinations = [[100, 1000], [200, 1000],
    #                 [100, 50000], [500, 50000], [1000, 50000], [5000, 50000],
    #                 [500, 100000], [1000, 100000], [5000, 100000], [10000, 100000]]

    # calc = []
    # max_ctgs = [20, 50, 100]

    # for nv, nc in combinations:
    #     for m in max_ctgs:
    #         calc.append([nv, nc, m, count_max_mislabels(nv, nc, m)])

    calc = pd.DataFrame(calc, columns=['n_virus', 'n_contigs', 'max_ctg_per_virus', 'False_labeling_rate'])

    g = sns.catplot(col='max_ctg_per_virus', y='False_labeling_rate', x='n_contigs', hue='n_virus',
                    data=calc, sharex=False, kind='bar')

    for t in g._legend.texts:
        new_label = "{:,}".format(int(t.get_text()))
        t.set_text(new_label)
        #t.set_text(str(t.get_text()))
    
    plt.show()
