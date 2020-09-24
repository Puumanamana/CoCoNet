'''
Configuration object to handle CLI, loading/resuming runs
'''

from pathlib import Path
from math import ceil
import logging
import shutil

import yaml
import h5py

from coconet.tools import kmer_count
from coconet.core.composition_feature import CompositionFeature
from coconet.core.coverage_feature import CoverageFeature
from coconet.log import setup_logger

class Configuration:
    """
    Configuration object to handle command line arguments
    """

    def __init__(self):
        self.io = {}
        self.cov_type = '.bam'
        self.features = ['coverage', 'composition']
        self.verbosity = 'INFO'

    def log(self, msg, level):
        try:
            logger = setup_logger('CoCoNet', self.io['log'], self.loglvl)
        except (KeyError, AttributeError):
            logger = logging.getLogger('CoCoNet')
        getattr(logger, level)(msg)

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

    def init_config(self, **kwargs):
        '''
        Make configuration from CLI
        '''

        for (name, value) in kwargs.items():
            if name not in {'fasta', 'h5', 'bam', 'tmp_dir', 'output'}:
                setattr(self, name, value)
            else:
                self.set_input(name, value)

        self.init_values_if_not_set()

        self.io['output'].mkdir(exist_ok=True)
        self.set_outputs()

    def set_input(self, name, val):
        '''
        Check inputs are well formatted before setting them
        '''

        if name == 'fasta':
            filepath = Path(val)
            if filepath.suffix not in ['.fa', '.fasta', '.fna']:
                self.log(f'Unknown file extension: {filepath.suffix}', 'critical')
                raise NotImplementedError

        if name == 'bam':
            if not val:
                return
            filepath = [Path(cov) for cov in val]
            suffixes = {cov.suffix for cov in filepath if cov != 'bam'}
            if not suffixes:
                self.log(f'Unknown file extension: {suffixes}', 'critical')
                raise NotImplementedError

        if name == 'h5':
            if val is None:
                return
            filepath = Path(val)
            self.cov_type = '.h5'

        if name in ['tmp_dir', 'output']:
            filepath = Path(val).resolve()

        self.io[name] = filepath

    def set_outputs(self):
        '''
        Define output file path for all steps
        '''

        output_files = dict(
            log='CoCoNet.log',
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
                self.log(f'h5 was set as input but the file does not exist', 'critical')
                raise FileNotFoundError

                if 'bam' not in self.io:
                    self.log(f'Could not find any bam file in the inputs. Aborting', 'critical')
                    raise FileNotFoundError

            elif not dest.is_file():
                shutil.copy(str(src), str(dest))

        if hasattr(self, 'theta'):
            output_files.update(dict(
                pre_graph='pre_graph.pkl',
                graph=f'graph_{self.theta}-{self.gamma1}-{self.gamma2}.pkl',
                assignments=f'bins_{self.theta}-{self.gamma1}-{self.gamma2}.csv'
            ))

        for name, filename in output_files.items():
            if isinstance(filename, str):
                output_files[name] = Path(self.io['output'], filename)
            if isinstance(filename, dict):
                for key, val in filename.items():
                    output_files[name][key] = Path(self.io['output'], val)

        self.io.update(output_files)

    def init_values_if_not_set(self):
        if not hasattr(self, 'fragment_length') or self.fragment_length < 0:
            if not hasattr(self, 'min_ctg_len'):
                self.min_ctg_len = 2048
            self.fragment_length = self.min_ctg_len // 2

        if not hasattr(self, 'min_ctg_len'):
            self.min_ctg_len = 2*self.fragment_length

    def to_yaml(self):
        '''
        Save configuration to YAML file
        '''

        to_save = self.__dict__
        config_file = Path(self.io['output'], 'config.yaml')

        if config_file.is_file() and config_file.stat().st_size > 0:
            complete_conf = Configuration.from_yaml(config_file).__dict__
            complete_conf.update(to_save)
        else:
            complete_conf = {key: val for (key, val) in self.__dict__.items()}

        io_to_keep = {k: v for (k, v) in self.io.items() if k in
                      {'fasta', 'h5', 'bam', 'tmp_dir', 'output'}}
        complete_conf['io'] = io_to_keep

        with open(config_file, 'w') as handle:
            yaml.dump(complete_conf, handle)

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

        return input_shapes.get(''.join(self.features), input_shapes)

    def get_architecture(self):
        '''
        Format neural network architecture
        '''

        architecture = dict(
            composition=dict(neurons=self.compo_neurons),
            coverage=dict(neurons=self.cover_neurons,
                           n_filters=self.cover_filters,
                           kernel_size=self.cover_kernel,
                           conv_stride=self.cover_stride),
            merge=dict(neurons=self.merge_neurons)
        )

        return architecture.get(''.join(self.features), architecture)

    def get_composition_feature(self):
        composition = CompositionFeature(
            path=dict(fasta=self.io['fasta'],
                      filt_fasta=self.io['filt_fasta'],
                      latent=self.io['repr'].get('composition', None))
        )
        if not composition.check_paths():
            self.log(
                ('Could not find the .fasta file. '
                 'Did you run coconet preprocess with the --fasta flag?'),
                'critical'
            )
            raise FileNotFoundError

        return composition

    def get_coverage_feature(self):
        coverage = CoverageFeature(
            path=dict(bam=self.io.get('bam', None),
                      h5=self.io.get('h5', None),
                      latent=self.io['repr'].get('coverage', None))
        )

        if not coverage.check_paths():
            self.log(
                ('Could not find the coverage information. '
                 'Did you run coconet preprocess with the --bam flag?'),
                'critical'
            )
            raise FileNotFoundError
        return coverage

    def get_features(self):
        features = []
        if 'coverage' in self.features:
            features.append(self.get_coverage_feature())
        if 'composition' in self.features:
            features.append(self.get_composition_feature())

        return features
