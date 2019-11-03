import os
from shutil import copyfile
from configparser import ConfigParser, ExtendedInterpolation, NoOptionError

from glob import glob
from math import ceil
import h5py
import numpy as np

class Experiment:
    '''
    Class to store experiment's parameters
    '''

    def __init__(self, name, root_dir='.',
                 in_dir='input_data', out_dir='output_data'):
        self.name = name
        self.indir = "{}/{}/{}".format(root_dir, in_dir, name)
        self.outdir = "{}/{}/{}".format(root_dir, out_dir, name)
        self.cfg = '{}/config.ini'.format(self.outdir)

        self.load(root_dir)
        self.set_io_files()

    def load(self, root_dir):
        '''
        Load configuration file
        '''

        if not os.path.exists(self.cfg):
            if not os.path.exists(self.outdir):
                os.mkdir(self.outdir)
        copyfile("{}/config.ini".format(root_dir), self.cfg)

        parser = ConfigParser(interpolation=ExtendedInterpolation())
        parser.read(self.cfg)

        self.threads = parser.getint('main', 'threads')
        self.model_type = parser.get('main', 'model_type')
        self.fl = parser.getint('main', 'fragment_length')
        self.min_ctg_len = self.fl * parser.getint('main', 'min_ctg_len_factor')
        self.step = self.fl // parser.getint('main', 'step_ratio')
        self.wsize = parser.getint('main', 'window_size')
        self.wstep = parser.getint('main', 'window_step')
        self.kmer = parser.getint('main', 'kmer')
        self.rc = parser.getboolean('main', 'rc')
        self.norm = parser.getboolean('main', 'norm')

        self.bam_processing = {
            'flag': parser.get('bam_processing', 'flag'),
            'min_qual': parser.get('bam_processing', 'min_mapping_quality'),
            'min_prevalence': parser.getint('bam_processing', 'min_prevalence_for_binning'),
            'fl_range': parser.get('bam_processing', 'fragment_length_range').split('-')}

        self.n_examples = {'train': parser.getint('training', 'n_examples_train'),
                           'test': parser.getint('training', 'n_examples_test')}

        self.train = {'batch_size': parser.getint('training', 'batch_size'),
                      'learning_rate': parser.getfloat('training', 'learning_rate'),
                      'load_batch': parser.getint('training', 'load_batch')}

        self.arch = {'composition': {'neurons': np.fromstring(
            parser.get('architecture', 'neurons_compo'), sep=',', dtype=int
        )},
                     'coverage': {'neurons': np.fromstring(
                         parser.get('architecture', 'neurons_cover'), sep=',', dtype=int),
                                  'n_filters': parser.getint('architecture', 'cover_filters'),
                                  'kernel_size': parser.getint('architecture', 'cover_kernel_size'),
                                  'conv_stride': parser.getint('architecture', 'cover_stride')},
                     'combination': {'neurons': np.fromstring(
                         parser.get('architecture', 'neurons_mixed'), sep=',', dtype=int
                     )}}
        self.clustering = {'n_frags': parser.getint('clustering', 'n_frags'),
                           'max_neighbors': parser.getint('clustering', 'max_neighbors'),
                           'hits_threshold': parser.getfloat('clustering', 'hits_threshold'),
                           'algo': parser.get('clustering', 'clustering_algorithm'),
                           }
        # For backward compatibility
        try:
            gamma_1 = parser.getfloat('clustering', 'clustering_gamma_1')
            gamma_2 = parser.getfloat('clustering', 'clustering_gamma_2')
        except NoOptionError:
            gamma_1 = 0.5
            gamma_2 = 0.05
        self.clustering['gamma_1'] = gamma_1
        self.clustering['gamma_2'] = gamma_2

    def set_io_files(self):
        '''
        Set file names for input and outputs
        '''

        self.inputs = {
            'raw': {
                'fasta': '{}/assembly.fasta'.format(self.indir),
                'coverage_h5': '{}/coverage_contigs.h5'.format(self.indir),
                'bam': sorted(glob('{}/*.bam'.format(self.indir)))
            },
            'filtered': {
                'fasta': '{}/assembly_gt{}.fasta'.format(self.indir, self.min_ctg_len),
                'coverage_h5': '{}/coverage_contigs_gt{}.h5'.format(self.indir, self.min_ctg_len)
            }
        }

        self.outputs = {
            'fragments': {'test': '{}/pairs_test.npy'.format(self.outdir),
                          'train': '{}/pairs_train.npy'.format(self.outdir)},
            'net': {'model': '{}/{}.pth'.format(self.outdir, self.model_type),
                    'test': '{}/{}_pred_test.csv'.format(self.outdir, self.model_type)},
            'repr': {
                'composition': '{}/representation_compo_nf{}.h5'.format(self.outdir, self.clustering['n_frags']),
                'coverage': '{}/representation_cover_nf{}.h5'.format(self.outdir, self.clustering['n_frags'])
            },
            'clustering': {
                'adjacency_matrix': '{}/adjacency_matrix_nf{}.npy'.format(
                    self.outdir, self.clustering['n_frags']),
                'refined_adjacency_matrix': '{}/adjacency_matrix_nf{}_refined.npy'.format(
                    self.outdir, self.clustering['n_frags']),
                'assignments': '{}/{}-{}_nf{}.csv'.format(
                    self.outdir, self.clustering['algo'], self.clustering['gamma_1'], self.clustering['n_frags']),
                'refined_assignments': '{}/{}-{}-{}_nf{}_refined.csv'.format(
                    self.outdir, self.clustering['algo'],
                    self.clustering['gamma_1'], self.clustering['gamma_2'],
                    self.clustering['n_frags'])}
        }

    def set_input_shapes(self):
        '''
        Calculate input shapes from loaded parameters
        '''

        h5_cov = h5py.File(self.inputs['filtered']['coverage_h5'], 'r')
        n_samples = h5_cov.get(list(h5_cov.keys())[0]).shape[0]
        h5_cov.close()

        self.input_shapes = {
            'composition': 4**self.kmer // (1+self.rc), # [sum([4**k // (1+self.rc) for k in self.kmer_list])],
            'coverage': (
                ceil((self.fl-self.wsize+1) / self.wstep),
                n_samples)
        }
