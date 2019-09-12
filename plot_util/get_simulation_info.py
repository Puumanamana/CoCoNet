import os,sys

import h5py
import numpy as np
from progressbar import progressbar

PARENT_DIR = os.path.join(sys.path[0], '..')

def get_coverage(coverage, n_samples):
    '''
    Calculates the coverage per contig in the simulation
    '''

    folder = "{}/input_data/{}_{}".format(PARENT_DIR, coverage, nsamples)

    h5file = h5py.File('{}/coverage_contigs.h5'.format(folder), 'r')
    contigs = list(h5file.keys())

    total_coverage = np.zeros((len(h5file), h5file.get(contigs[0]).shape[0]))

    for i, ctg in progressbar(enumerate(h5file),
                              max_value=len(h5file)):
        total_coverage[i] = np.mean(h5file.get(ctg)[:], axis=1)

    print('''
    In theory: n_samples = {} - coverage = {}
    Observed: n_samples = {} - coverage = {}'''
          .format(n_samples, coverage,
                  total_coverage.shape[1],
                  total_coverage.mean(axis=0)))
    return total_coverage

if __name__ == '__main__':
    # import argparse

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--coverage', type=int)
    # parser.add_argument('--nsamples', type=int)

    # args = parser.parse_args()

    for nsamples in [3, 5, 10]:
        for coverage in [1, 2, 5, 10]:
            get_coverage(coverage, nsamples)
