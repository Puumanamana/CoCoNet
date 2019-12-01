import os
from shutil import copyfile
import re
# from configparser import ConfigParser, ExtendedInterpolation, NoOptionError

from glob import glob
from math import ceil
import h5py
import numpy as np

class Experiment:
    '''
    Class to store experiment's parameters
    '''

    def __init__(self, name, threads, args):
        self.name = name
        self.threads = threads
        self.model_type = 'CoCoNet'

        self.load(args)
        self.set_io_files()
        self.input_shapes = {}

    def load(self, args):
        self.io = args['I/O']
        self.frag = args['Fragmentation']
        self.bam = args['Bam processing']
        self.dl = args['Deep learning']
        self.cluster = args['Clustering']

        self.arch = {
            'composition': {'neurons': self.dl['compo_neurons']},
            'coverage': {'neurons': self.dl['cover_neurons'],
                         'n_filters': self.dl['cover_filters'],
                         'kernel_size': self.dl['cover_kernel_size'],
                         'conv_stride': self.dl['cover_stride']},
            'combination': {'neurons': self.dl['neurons_mixed']}
        }

    def set_io_files(self):
        '''
        Set file names for input and outputs
        '''

        self.inputs = {
            'raw': {
                'fasta': '{}/assembly.fasta'.format(self.io['input']),
                'coverage_h5': '{}/coverage_contigs.h5'.format(self.io['input']),
                'bam': sorted([bam for bam in glob('{}/*.bam'.format(self.io['input']))
                               if not re.match('.*fl.*_sorted.bam', bam)])
            },
            'filtered': {
                'fasta': '{}/assembly_gt{}_prev{}.fasta'.format(self.io['input'], self.bam['min_ctg_len'], self.bam['min_prevalence']),
                'coverage_h5': '{}/coverage_contigs_gt{}.h5'.format(self.io['input'], self.bam['min_ctg_len'])
            }
        }

        self.outputs = {
            'singletons': "{}/singletons.txt".format(self.io['output']),
            'fragments': {'test': '{}/pairs_test.npy'.format(self.io['output']),
                          'train': '{}/pairs_train.npy'.format(self.io['output'])},
            'net': {'model': '{}/{}.pth'.format(self.io['output'], self.model_type),
                    'test': '{}/{}_pred_test.csv'.format(self.io['output'], self.model_type)},
            'repr': {
                'composition': '{}/representation_compo_nf{}.h5'.format(self.io['output'], self.cluster['n_frags']),
                'coverage': '{}/representation_cover_nf{}.h5'.format(self.io['output'], self.cluster['n_frags'])
            },
            'clustering': {
                'adjacency_matrix': '{}/adjacency_matrix_nf{}.npy'.format(
                    self.io['output'], self.cluster['n_frags']),
                'refined_adjacency_matrix': '{}/adjacency_matrix_nf{}_refined.npy'.format(
                    self.io['output'], self.cluster['n_frags']),
                'assignments': '{}/leiden-{}-{}_nf{}.csv'.format(
                    self.io['output'], self.cluster['hits_threshold'],
                    self.cluster['gamma1'], self.cluster['n_frags']),
                'refined_assignments': '{}/leiden-{}-{}-{}_nf{}_refined.csv'.format(
                    self.io['output'], self.cluster['hits_threshold'],
                    self.cluster['gamma1'], self.cluster['gamma2'],
                    self.cluster['n_frags'])}
        }

    def set_input_shapes(self):
        '''
        Calculate input shapes from loaded parameters
        '''

        h5_cov = h5py.File(self.inputs['filtered']['coverage_h5'], 'r')
        n_samples = h5_cov.get(list(h5_cov.keys())[0]).shape[0]
        h5_cov.close()

        self.input_shapes = {
            'composition': 4**self.dl['kmer'] // (1+self.dl['rc']),
            'coverage': (
                ceil((self.frag['fl']-self.dl['wsize']+1) / self.dl['wstep']),
                n_samples)
        }
