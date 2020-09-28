'''
Unittest for Configuration object
'''

from pathlib import Path
import shutil

from coconet.core.config import Configuration

from .data import generate_h5_file

def test_init():
    '''
    Test __init__ method for Configuration object
    '''

    cfg1 = Configuration()

    assert hasattr(cfg1, 'io')

def test_init_config():
    '''
    Test init_config method for Configuration object
    '''

    kwargs = {'a': 1, 'fasta': '/a/b/c.fasta', 'output': 'test123abc'}

    cfg = Configuration()
    cfg.init_config(**kwargs)

    assert cfg.a == kwargs['a']
    assert cfg.io['fasta'] == Path(kwargs['fasta'])
    assert cfg.io['output'].is_dir()

    shutil.rmtree(cfg.io['output'])

def test_load_save():
    '''
    Test saving / loading configs
    '''

    kwargs = {'h5': Path('xyz.h5'),
              'fasta': Path('/a/b/c.fasta'),
              'output': Path('output')}

    kwargs['h5'].touch()

    cfg = Configuration()
    cfg.init_config(**kwargs)
    cfg.to_yaml()

    config_file = Path(cfg.io['output'], 'config.yaml')

    cfg_loaded = Configuration.from_yaml(config_file)

    cfg_is_created = config_file.is_file()
    attrs_exist = all(getattr(cfg_loaded, k) == getattr(cfg, k)
                      for k, v in cfg_loaded.__dict__.items()
                      if not isinstance(v, dict))

    kwargs['h5'].unlink()
    shutil.rmtree(cfg.io['output'])

    assert cfg_is_created
    assert attrs_exist

def test_input_sizes():
    '''
    Check if input sizes are correct
    '''

    cfg = Configuration()
    cfg.init_config(output='test123', kmer=4, no_rc=False,
                    fragment_length=10, wsize=4, wstep=2)
    cfg.io['h5'] = generate_h5_file(10, filename='coverage.h5')

    input_shapes = {'composition': 136,
                    'coverage': (4, 2)}

    auto_shapes = cfg.get_input_shapes()

    shutil.rmtree(cfg.io['output'])
    
    assert input_shapes == auto_shapes

def test_architecture():
    '''
    Test if params for architecture are formatted correctly
    '''

    args = {'compo_neurons': [64, 32],
            'cover_neurons': [32, 32],
            'cover_filters': 30,
            'cover_kernel': 5,
            'cover_stride': 2,
            'merge_neurons': 16}

    cfg = Configuration()
    cfg.init_config(output='test123', **args)

    architecture = {
        'composition': {'neurons': args['compo_neurons']},
        'coverage': {'neurons': args['cover_neurons'],
                     'n_filters': args['cover_filters'],
                     'kernel_size': args['cover_kernel'],
                     'conv_stride': args['cover_stride']},
        'merge': {'neurons': args['merge_neurons']}
    }

    shutil.rmtree(cfg.io['output'])
    
    assert architecture == cfg.get_architecture()

if __name__ == '__main__':
    test_load_save()
