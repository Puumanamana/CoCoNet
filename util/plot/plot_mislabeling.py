from scipy.special import comb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def count_max_mislabels(n_vir, n_ctg, max_ctg_per_virus):
    '''
    Counts maximum mislabels
    '''
    false_negatives = comb(max_ctg_per_virus, 2) * n_vir
    total_negatives = comb(n_ctg, 2)

    return false_negatives / total_negatives

def main():
    '''
    '''
    calc = []

    for n_vir in [500, 1000, 5000, 10000]:
        for nc_factor in [10, 20]:
            n_ctg = nc_factor*n_vir
            for m in [20, 50, 100]:
                # if m >= 10 * n_ctg / n_vir:
                #     continue
                calc.append([n_vir, n_ctg, m, count_max_mislabels(n_vir, n_ctg, m)])

    calc = pd.DataFrame(calc, columns=['n_virus', 'n_contigs', 'max_ctg_per_virus', 'False_labeling_rate'])

    graph = sns.catplot(x='n_contigs', y='False_labeling_rate',
                        hue='n_virus', col='max_ctg_per_virus',
                        data=calc, sharex=False, kind='bar')

    for txt in graph._legend.texts:
        new_label = "{:,}".format(int(txt.get_text()))
        txt.set_text(new_label)

    plt.show()


if __name__ == '__main__':
    main()
