'''
Unittest for Configuration object
'''

from pathlib import Path

from coconet.config import Configuration

from .data import generate_h5_file

def test_init():
    '''
    Test __init__ method for Configuration object
    '''

    cfg1 = Configuration()
    cfg2 = Configuration(a=1, output='abc')

    assert hasattr(cfg1, 'io')
    assert cfg2.a == 1
    assert cfg2.io['output'] == Path('abc').resolve()

def test_init_config():
    '''
    Test init_config method for Configuration object
    '''

    kwargs = {'a': 1, 'fasta': '/a/b/c.fasta', 'output': 'test123abc'}

    cfg = Configuration()
    cfg.init_config(**kwargs, mkdir=True)

    assert cfg.a == kwargs['a']
    assert cfg.io['fasta'] == Path(kwargs['fasta'])
    assert cfg.io['output'].is_dir()

    cfg.io['output'].rmdir()

def test_load_save():
    '''
    Test saving / loading configs
    '''

    kwargs = {'coverage': ['xyz.h5'],
              'fasta': '/a/b/c.fasta',
              'output': 'output'}

    cfg = Configuration()
    cfg.init_config(mkdir=True, **kwargs)
    cfg.to_yaml()

    config_file = Path('{}/config.yaml'.format(cfg.io['output']))

    cfg_loaded = Configuration.from_yaml(config_file)

    assert config_file.is_file()
    assert all(getattr(cfg_loaded, k) == getattr(cfg, k)
               for k, v in cfg_loaded.__dict__.items()
               if not isinstance(v, dict))

    config_file.unlink()
    cfg.io['output'].rmdir()

def test_input_sizes():
    '''
    Check if input sizes are correct
    '''

    cfg = Configuration()
    cfg.init_config(output='test123', kmer=4, no_rc=False, fragment_length=10, wsize=4, wstep=2)
    cfg.io['filt_h5'] = generate_h5_file(10)

    input_shapes = {'composition': 136,
                    'coverage': (4, 2)}

    auto_shapes = cfg.get_input_shapes()

    cfg.io['filt_h5'].unlink()

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

    assert architecture == cfg.get_architecture()
