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

    print('Fetching coverage for {}'.format(h5_path))
    h5file = h5py.File(h5_path, 'r')
    contigs = list(h5file.keys())

    total_coverage = np.zeros((len(h5file), h5file.get(contigs[0]).shape[0], 2))
    prevalence = np.zeros(len(h5file))

    for i, ctg in progressbar(enumerate(h5file),
                              max_value=len(h5file)):
        cov = h5file.get(ctg)[:]
        total_coverage[i, :, 0] = np.mean(cov, axis=1)
        total_coverage[i, :, 1] = np.std(cov, axis=1)
        prevalence[i] = (cov.sum(axis=1) > 0).sum()

    return total_coverage, prevalence

def fformat(x):
    return "{:.2f}".format(x)

def process_sim(n_genomes, coverage, n_samples, name):

    if name == '':
        name = "{}_{}_{}".format(n_genomes, coverage, n_samples)

    cfg = Experiment(name)

    ctg_info = pd.DataFrame({
        seq.id: [len(seq.seq), seq.id.split('|')[0]]
        for seq in SeqIO.parse(cfg.inputs['filtered']['fasta'], 'fasta')
    }, index=['length', 'virus']).T

    bin_info = ctg_info.groupby('virus').length.agg(len)

    (real_coverage, prevalence) = get_coverage(cfg.inputs['filtered']['coverage_h5'])

    summary = {
        'n_samples': n_samples,
        'simulated_coverage': coverage,
        'real_coverage_mean': np.array2string(real_coverage[:, :, 0].mean(axis=0),
                                              formatter={'float_kind': fformat}),
        'real_coverage_std': np.array2string(real_coverage[:, :, 1].mean(axis=0),
                                             formatter={'float_kind': fformat}),
        'contig_length': "{:}, {:}, {:}".format(ctg_info.length.min(),
                                                ctg_info.length.median(),
                                                ctg_info.length.max()),
        'n_contigs': len(ctg_info),
        'n_bins': n_genomes,
        'bin_size': (bin_info
                     .describe()
                     .drop(['count', '25%', '75%', 'min'])
                     .apply(fformat)
                     .to_dict()),
        'mean prevalence': prevalence.mean()
    }

    return summary

def main():
    '''
    Print summmary of given dataset
    '''

    args = parse_args()

    if args.iter > 0:
        summary = process_sim(args.n_genomes, args.coverage, args.nsamples, args.name)
        print("\n".join(['{}: {}'.format(k, str(v)) for k, v in summary.items()]))
        return summary

    summaries = pd.Series([process_sim(args.n_genomes, args.coverage, args.nsamples,
                           '{}_{}_{}_{}'.format(args.n_genomes, args.coverage, args.nsamples, iter_nb))
                           for iter_nb in range(10)])
    
    (summaries
     .drop(['real_coverage_mean', 'real_coverage_std', 'contig_length'], axis=1)
     .groupby(['n_bins', 'simulated_coverage', 'n_samples'])
     .agg(lambda x: '{} (+/- {})'.format(x.mean(), x.std()))
     .to_csv("simulation_summary.csv"))

if __name__ == '__main__':
    main()
