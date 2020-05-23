'''
Configuration object to handle CLI, loading/resuming runs
'''

import sys
from pathlib import Path
from math import ceil

import yaml
import h5py

from coconet.tools import kmer_count

class Configuration:
    """
    Configuration object to handle command line arguments
    """

    def __init__(self, **kwargs):
        self.io = {}
        self.cov_type = '.bam'

        if kwargs:
            for item in kwargs.items():
                self.set_input(*item)

    @classmethod
    def from_yaml(cls, filepath):
        '''
        Load config from saved file
        '''

        config = Configuration()

        with open(str(filepath), 'r') as handle:
            kwargs = yaml.load(handle, Loader=yaml.FullLoader)

        # Reset output in case the folder was moved
        kwargs['io']['output'] = Path(filepath).parent
        config.init_config(**kwargs)

        return config

    def init_config(self, mkdir=False, **kwargs):
        '''
        Make configuration from CLI
        '''

        for item in kwargs.items():
            self.set_input(*item)

        self.set_outputs(mkdir)

    def set_input(self, name, val):
        '''
        Check inputs are well formatted before setting them
        '''

        if name not in {'fasta', 'coverage', 'tmp_dir', 'output'}:
            setattr(self, name, val)
            return

        if name == 'fasta':
            filepath = Path(val)
            if filepath.suffix not in ['.fa', '.fasta', '.fna']:
                sys.exit('This assembly file extension is not supported ({})'.format(filepath.suffix))

        if name == 'coverage':
            filepath = [Path(cov) for cov in val]
            if len(filepath) == 1:
                filepath = filepath[0]
                self.cov_type = '.h5'
                name = 'coverage_h5'
            else:
                suffixes = {cov.suffix for cov in filepath if cov != 'bam'}
                if not suffixes:
                    sys.exit('This coverage file extension is not supported ({})'.format(suffixes))
                name = 'coverage_bam'

        if name in ['tmp_dir', 'output']:
            filepath = Path(val).resolve()

        self.io[name] = filepath


    def to_yaml(self):
        '''
        Save configuration to YAML file
        '''

        to_save = self.__dict__
        config_file = Path('{}/config.yaml'.format(self.io['output']))

        if config_file.is_file() and config_file.stat().st_size > 0:
            complete_conf = Configuration.from_yaml(config_file).__dict__
            complete_conf.update(to_save)
        else:
            complete_conf = {key: val for (key, val) in self.__dict__.items()}

        io_to_keep = {k: v for (k, v) in self.io.items() if k in ['fasta', 'coverage', 'tmp_dir', 'output']}
        complete_conf['io'] = io_to_keep

        with open(config_file, 'w') as handle:
            yaml.dump(complete_conf, handle)

    def set_outputs(self, mkdir=False):
        '''
        Define output file path for all steps
        '''

        if mkdir:
            self.io['output'].mkdir(exist_ok=True)

        output_files = {
            'filt_fasta': 'assembly_filtered.fasta',
            'filt_h5': 'coverage_filtered.h5',
            'singletons': 'singletons.txt',
            'pairs': {'test': 'pairs_test.npy', 'train': 'pairs_train.npy'},
            'model': 'CoCoNet.pth',
            'nn_test': 'CoCoNet_test.csv',
            'repr': {'composition': 'representation_compo.h5',
                     'coverage': 'representation_cover.h5'}
        }

        if 'theta' in self.__dict__:
            output_files.update({
                'pre_graph': 'pre_graph.pkl',
                'graph': 'graph_{}-{}-{}.pkl'.format(
                    self.theta, self.gamma1, self.gamma2),
                'assignments': 'bins_{}-{}-{}.csv'.format(
                    self.theta, self.gamma1, self.gamma2)
            })

        for name, filename in output_files.items():
            if isinstance(filename, str):
                output_files[name] = Path('{}/{}'.format(self.io['output'], filename))
            if isinstance(filename, dict):
                for key, val in filename.items():
                    output_files[name][key] = Path('{}/{}'.format(self.io['output'], val))

        self.io.update(output_files)

    def get_input_shapes(self):
        '''
        Return input shapes for neural network
        '''

        with h5py.File(self.io['filt_h5'], 'r') as handle:
            n_samples = handle.get(list(handle.keys())[0]).shape[0]

        input_shapes = {
            'composition': kmer_count(self.kmer, rc=not self.no_rc),
            'coverage': (ceil((self.fragment_length-self.wsize+1) / self.wstep), n_samples)
        }

        return input_shapes

    def get_architecture(self):
        '''
        Format neural network architecture
        '''

        architecture = {
            'composition': {'neurons': self.compo_neurons},
            'coverage': {'neurons': self.cover_neurons,
                         'n_filters': self.cover_filters,
                         'kernel_size': self.cover_kernel,
                         'conv_stride': self.cover_stride},
            'merge': {'neurons': self.merge_neurons}
        }

        return architecture
