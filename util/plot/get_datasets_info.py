import os
import sys
import argparse
from progressbar import progressbar

import numpy as np
import pandas as pd
import h5py
from Bio import SeqIO

PARENT_DIR = os.path.join(sys.path[0], '../..')
sys.path.insert(1, PARENT_DIR)
from experiment import Experiment

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_genomes', type=int, default=-1)
    parser.add_argument('--coverage', type=int, default=-1)
    parser.add_argument('--nsamples', type=int, default=-1)
    parser.add_argument('--iter', type=int, default=-1)
    parser.add_argument('--name', type=str, default='')

    args = parser.parse_args()

    if args.name.replace('_', '').isdigit() and args.name.count('_') == 3:
        (args.n_genomes, args.coverage, args.nsamples, args.iter) = args.name.split('_')

    return args

def get_coverage(h5_path):
    '''
    Calculates the coverage per contig in the simulation
    '''

    print('\033[1mFetching coverage for {}\033[0m'.format(h5_path))
    h5file = h5py.File(h5_path, 'r')
    contigs = list(h5file.keys())

    total_coverage = np.zeros((len(h5file), h5file.get(contigs[0]).shape[0], 2))
    prevalence = np.zeros(len(h5file))
    cov_sum, len_sum = (0, 0)

    for i, ctg in progressbar(enumerate(h5file),
                              max_value=len(h5file)):
        cov = h5file.get(ctg)[:]
        total_coverage[i, :, 0] = np.mean(cov, axis=1)
        total_coverage[i, :, 1] = np.std(cov, axis=1)
        prevalence[i] = (cov.sum(axis=1) > cov.shape[1]/10).sum()

        cov_sum += cov.sum(axis=1)
        len_sum += cov.shape[1]

    return total_coverage, prevalence, cov_sum / len_sum

def fformat(x):
    if isinstance(x, str):
        if x.isdigit():
            return fformat(float(x))
        return x

    if pd.isnull(x):
        return x

    if float(x) == int(x):
        return "{:g}".format(int(x))
    return "{:.1f}".format(x)

def process_sim(n_genomes, coverage, n_samples, name):

    if name == '':
        name = "{}_{}_{}".format(n_genomes, coverage, n_samples)

    cfg = Experiment(name)

    ctg_info = pd.DataFrame({
        seq.id: [len(seq.seq), seq.id.split('|')[0]]
        for seq in SeqIO.parse(cfg.inputs['filtered']['fasta'], 'fasta')
    }, index=['length', 'virus']).T

    bin_info = ctg_info.groupby('virus').length.agg(len)

    (real_coverage, prevalence, xcov) = get_coverage(cfg.inputs['filtered']['coverage_h5'])

    if n_samples == -1:
        n_samples = 3
        coverage = '{} (+/- {})'.format(fformat(xcov.mean()), fformat(xcov.std()))
        n_genomes = '>1500'

    print(coverage, xcov)

    summary = {
        'Number of bins': n_genomes,
        'Coverage': coverage,
        'Number of samples': n_samples,
        'Number of contigs': len(ctg_info),
        'Bin_sizes': bin_info.mean(),
        'Prevalence_mean': prevalence.mean(),
        'Prevalence_std': prevalence.std(),
        'Real_coverage_mean': np.array2string(real_coverage[:, :, 0].mean(axis=0),
                                              formatter={'float_kind': fformat}),
        'Real_coverage_std': np.array2string(real_coverage[:, :, 1].mean(axis=0),
                                             formatter={'float_kind': fformat}),
        'Contig length': "{:}, {:}, {:}".format(ctg_info.length.min(),
                                                ctg_info.length.median(),
                                                ctg_info.length.max()),
    }

    return summary

def main():
    '''
    Print summmary of given dataset
    '''

    args = parse_args()

    if args.iter > 0 or args.name:
        summary = process_sim(args.n_genomes, args.coverage, args.nsamples, args.name)
        print("\n".join(['{}: {}'.format(k, str(v)) for k, v in summary.items()]))
        return summary

    summaries = [process_sim(-1, -1, -1, "Station_Aloha")]

    summaries += [process_sim(
        n_genomes, coverage, n_samples, f"{n_genomes}_{coverage}_{n_samples}_{iter_nb}")
                  for n_genomes in [500, 2000]
                  for coverage in [3, 10]
                  for n_samples in [4, 15]
                  for iter_nb in range(10)]

    result = (pd.DataFrame(summaries)
              .drop(['Real_coverage_mean', 'Real_coverage_std', 'Contig length'], axis=1)
              .groupby(['Number of bins', 'Coverage', 'Number of samples'])
              .agg(lambda x: '{} (+/- {})'.format(fformat(x.mean()), fformat(x.std())))
              .sort_index())

    for col in result.columns:
        result[col] = result[col].str.replace(' (+/- nan)', '', regex=False)

    prevalence = []
    for _, row in result.iterrows():
        prev_mu = fformat(row.Prevalence_mean.split()[0])
        prev_sig = fformat(row.Prevalence_std.split()[0])
        prevalence.append("{} (+/- {})".format(prev_mu, prev_sig))

    result['Prevalence'] = prevalence

    result = result.reset_index().drop(['Prevalence_mean', 'Prevalence_std'], axis=1)
    result.Coverage = result.Coverage.astype(str) + 'X'
    result.index = ['Sim-{}'.format(i+1) for i in result.index[:-1]] + ['Station Aloha']
    result.index.name = 'Dataset'

    result.to_csv("simulation_summary.csv")

if __name__ == '__main__':
    main()
