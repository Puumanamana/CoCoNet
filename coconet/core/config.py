'''
Configuration object to handle CLI, loading/resuming runs
'''

import sys
from pathlib import Path
from math import ceil
import logging

import yaml
import h5py

from coconet.tools import kmer_count
from coconet.core.composition_feature import CompositionFeature
from coconet.core.coverage_feature import CoverageFeature


class Configuration:
    """
    Configuration object to handle command line arguments
    """

    def __init__(self):
        self.io = {}
        self.cov_type = '.bam'
        self.features = ['coverage', 'composition']
        self.logger = None
        self.verbosity = 'INFO'

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

        if mkdir:
            self.io['output'].mkdir(exist_ok=True)

        if self.logger is None:
            self.set_logging()

        self.set_outputs()

    def set_input(self, name, val):
        '''
        Check inputs are well formatted before setting them
        '''

        if name not in {'fasta', 'h5', 'bam', 'tmp_dir', 'output'}:
            setattr(self, name, val)
            return

        if name == 'fasta':
            filepath = Path(val)
            if filepath.suffix not in ['.fa', '.fasta', '.fna']:
                sys.exit('This assembly file extension is not supported ({})'.format(filepath.suffix))

        if name == 'bam':
            if not val:
                return
            filepath = [Path(cov) for cov in val]
            suffixes = {cov.suffix for cov in filepath if cov != 'bam'}
            if not suffixes:
                sys.exit('This coverage file extension is not supported ({})'.format(suffixes))

        if name == 'h5':
            if val is None:
                return
            filepath = Path(val)
            self.cov_type = '.h5'

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

        if 'logger' in complete_conf:
            del complete_conf['logger']

        io_to_keep = {k: v for (k, v) in self.io.items() if k in
                      {'fasta', 'h5', 'bam', 'tmp_dir', 'output'}}
        complete_conf['io'] = io_to_keep

        with open(config_file, 'w') as handle:
            yaml.dump(complete_conf, handle)

    def set_outputs(self):
        '''
        Define output file path for all steps
        '''

        output_files = dict(
            filt_fasta='assembly_filtered.fasta',
            h5='coverage.h5',
            singletons='singletons.txt',
            pairs={'test': 'pairs_test.npy', 'train': 'pairs_train.npy'},
            model='CoCoNet.pth',
            nn_test='CoCoNet_test.csv',
            repr={feature: f'latent_{feature}.h5' for feature in self.features}
        )

        # if coverage_h5 already exists, symlink it to the output folder
        if 'h5' in self.io:
            src = self.io['h5'].resolve()
            dest = Path(self.io['output'], output_files['h5']).resolve()

            if not src.is_file():
                self.logger.warning(f'h5 was set as input but the file does not exist')
                if 'bam' not in self.io:
                    self.logger.error(f'Could not find any bam file in the inputs. Aborting')
                    sys.exit()

            elif not dest.is_file():
                dest.symlink_to(src)

        if hasattr(self, 'theta'):
            output_files.update({
                'pre_graph': 'pre_graph.pkl',
                'graph': 'graph_{}-{}-{}.pkl'.format(
                    self.theta, self.gamma1, self.gamma2),
                'assignments': 'bins_{}-{}-{}.csv'.format(
                    self.theta, self.gamma1, self.gamma2)
            })

        for name, filename in output_files.items():
            if isinstance(filename, str):
                output_files[name] = Path(self.io['output'], filename)
            if isinstance(filename, dict):
                for key, val in filename.items():
                    output_files[name][key] = Path('{}/{}'.format(self.io['output'], val))

        self.io.update(output_files)

    def set_logging(self):
        self.io['log'] = Path(self.io['output'], 'coconet.log')
        self.logger = logging.getLogger('CoCoNet')
        self.logger.setLevel('DEBUG')

        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)-8s : %(message)s',
            '%Y-%m-%d %H:%M:%S'
        )

        self.logger.propagate = False
        self.logger.setLevel('DEBUG')

        if not self.logger.handlers:
            stream_hdl = logging.StreamHandler()
            stream_hdl.setFormatter(formatter)
            stream_hdl.setLevel(self.verbosity)

            file_hdl = logging.FileHandler(str(self.io['log']))
            file_hdl.setFormatter(formatter)

            self.logger.addHandler(stream_hdl)
            self.logger.addHandler(file_hdl)


    def get_input_shapes(self):
        '''
        Return input shapes for neural network
        '''

        with h5py.File(self.io['h5'], 'r') as handle:
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

    def get_composition_feature(self):
        return CompositionFeature(
            path=dict(fasta=self.io['fasta'],
                      filt_fasta=self.io['filt_fasta'],
                      latent=self.io['repr']['composition'])
            )

    def get_coverage_feature(self):
        return CoverageFeature(
            path=dict(bam=self.io.get('bam', None),
                      h5=self.io.get('h5', None),
                      latent=self.io['repr']['coverage'])
        )

    def get_features(self):
        features = {}
        if 'coverage' in self.features:
            features['coverage'] = self.get_coverage_feature()
        if 'composition' in self.features:
            features['composition'] = self.get_composition_feature()

        return features
