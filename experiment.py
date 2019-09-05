from configparser import ConfigParser, ExtendedInterpolation
from glob import glob
import h5py
import numpy as np
from math import ceil

class Experiment:

    def __init__(self,name,root_dir='.'):
        self.name = name
        self.indir = "{}/input_data/{}".format(root_dir,name)
        self.outdir = "{}/output_data/{}".format(root_dir,name)
        self.cfg = '{}/config.ini'.format(root_dir)
        self.load()
        self.set_io_files()
        # self.cfg = 'input_data/{}/config.ini'.format(name)

    def set_io_files(self):
        self.inputs = {
            'raw': { 'fasta': '{}/assembly.fasta'.format(self.indir),
                     'coverage_h5': '{}/coverage_contigs.h5'.format(self.indir),
                     'bam': glob('{}/*.bam'.format(self.indir)) },
            'filtered': { 'fasta': '{}/assembly_gt{}.fasta'.format(self.indir,self.min_ctg_len),
                          'coverage_h5': '{}/coverage_contigs_gt{}.h5'.format(self.indir,self.min_ctg_len) }
        }

        self.outputs = {
            'fragments': { 'test': '{}/pairs_test.npy'.format(self.outdir),
                           'train': '{}/pairs_train.npy'.format(self.outdir) },
            'model': '{}/{}.pth'.format(self.outdir,self.model_type),
            'repr': { 'composition': '{}/representation_compo_nf{}.h5'.format(self.outdir,self.clustering['n_frags']),
                      'coverage': '{}/representation_cover_nf{}.h5'.format(self.outdir,self.clustering['n_frags'])}
        }
        
    def load(self):
        parser = ConfigParser(interpolation=ExtendedInterpolation())
        parser.read(self.cfg)

        self.model_type = parser.get('main','model_type')
        self.fl = parser.getint('main','fragment_length')
        self.min_ctg_len = self.fl * parser.getint('main','min_ctg_len_factor')
        self.step = self.fl // parser.getint('main','step_ratio')
        self.wsize = parser.getint('main','window_size')
        self.wstep = parser.getint('main','window_step')
        self.kmer_list = np.fromstring(parser.get('main','kmer_list'),sep=',',dtype=int)
        self.rc = parser.getboolean('main','rc')
        self.norm = parser.getboolean('main','norm')

        self.n_examples = { 'train': parser.getint('training','n_examples_train'),
                            'test': parser.getint('training','n_examples_test')}

        self.train = { 'batch_size': parser.getint('training','batch_size'),
                       'learning_rate': parser.getfloat('training','learning_rate'),
                       'load_batch': parser.getint('training','load_batch') }
        
        self.arch = { 'composition': {'neurons': np.fromstring(parser.get('architecture','neurons_compo'), sep=',',dtype=int) },
                      'coverage': {'neurons': np.fromstring(parser.get('architecture','neurons_cover'),sep=',',dtype=int),
                                   'n_filters': parser.getint('architecture','cover_filters'),
                                   'kernel_size': parser.getint('architecture','cover_kernel_size'),
                                   'conv_stride': parser.getint('architecture','cover_stride')},
                      'combination': {'neurons': np.fromstring(parser.get('architecture','neurons_mixed'),sep=',',dtype=int)}
        }
        self.clustering = { 'n_frags': parser.getint('clustering','n_frags'),
                            'max_neighbors': parser.getint('clustering','max_neighbors'),
                            'hits_threshold': parser.getfloat('clustering','hits_threshold') }

    def set_input_shapes(self):
        h5_cov = h5py.File(self.inputs['filtered']['coverage_h5'],'r')
        n_samples = h5_cov.get(list(h5_cov.keys())[0]).shape[0]
        h5_cov.close()
        
        self.input_shapes = {
            'composition': [ sum([ 4**k // (1+self.rc) for k in self.kmer_list ]) ],
            'coverage': (
                ceil((self.fl-self.wsize+1) / self.wstep),
                n_samples)
        }                    
