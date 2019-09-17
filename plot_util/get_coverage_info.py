import os,sys
from glob import glob

import h5py
import numpy as np
from progressbar import progressbar

PARENT_DIR = os.path.join(sys.path[0], '..')

def get_coverage(coverage=-1, n_samples=-1, path=None):
    '''
    Calculates the coverage per contig in the simulation
    '''

    if coverage > 0 and n_samples > 0:
        folder = "{}/input_data/{}_{}".format(PARENT_DIR, coverage, n_samples)
    else:
        folder = path

    coverage_file = glob('{}/coverage_contigs*.h5'.format(folder))[0]
    h5file = h5py.File(coverage_file, 'r')
    contigs = list(h5file.keys())

    total_coverage = np.zeros((len(h5file), h5file.get(contigs[0]).shape[0], 2))

    for i, ctg in progressbar(enumerate(h5file),
                              max_value=len(h5file)):
        cov = h5file.get(ctg)[:]
        total_coverage[i, :, 0] = np.mean(cov, axis=1)
        total_coverage[i, :, 1] = np.std(cov, axis=1)

    print('''
    Coverage file path: {}
    Simulated (n_samples = {}, coverage = {})
    Observed ({} samples): 
    coverage = 
    {}
    '''
          .format(coverage_file,
                  n_samples, coverage,
                  total_coverage.shape[1],
                  total_coverage.mean(axis=0),
          ))
    return total_coverage

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--coverage', type=int, default=-1)
    parser.add_argument('--nsamples', type=int, default=-1)
    parser.add_argument('--path', type=str, default='.')

    args = parser.parse_args()

    get_coverage(coverage=args.coverage, n_samples=args.nsamples, path=args.path)
