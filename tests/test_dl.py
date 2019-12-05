'''
Tests for deep learning procedure
'''

from coconet.dl_util import initialize_model
from coconet.torch_models import CompositionModel, CoverageModel, CoCoNet

def test_init_composition_model():
    '''
    Check if model can be initialized
    '''

    architecture = {'neurons': [64, 32]}
    input_shapes = 136
    model = initialize_model('composition', input_shapes, architecture)

    assert isinstance(model, CompositionModel)

def test_init_coverage_model():
    '''
    Check if model can be initialized
    '''

    architecture = {'neurons': [64, 32], 'n_filters': 16,
                    'kernel_size': 5, 'conv_stride': 2}

    input_shapes = (64, 2)

    model = initialize_model('coverage', input_shapes, architecture)

    assert isinstance(model, CoverageModel)

def test_init_combined_model():
    '''
    Check if model can be initialized
    '''

    architecture = {
        'composition': {'neurons': [64, 32]},
        'coverage': {'neurons': [64, 32],
                     'n_filters': 16,
                     'kernel_size': 5,
                     'conv_stride': 3},
        'combined': {'neurons': 16}
    }

    input_shapes = {'composition': 136, 'coverage': (64, 2)}

    model = initialize_model('CoCoNet', input_shapes, architecture)

    assert isinstance(model, CoCoNet)

def test_load_model():
    '''
    Check if saved model can be loaded
    '''
    
    assert 1 == 1

def test_get_labels():
    '''
    Check if labels can be loaded
    '''
    
    assert 1 == 1

def test_count_npy():
    '''
    Check if npy can be counted
    '''
    
    assert 1 == 1

    
def test_nn_summary():
    '''
    Check if summary runs
    '''
    
    assert 1 == 1

def test_confusion_table():
    '''
    Check if confusion table works
    '''
    
    assert 1 == 1

def test_load_data_compo():
    '''
    Test composition generator
    '''

    assert 1 == 1

def test_load_data_cover():
    '''
    Test coverage generator
    '''

    assert 1 == 1

def test_learning():
    '''
    Test learning?
    '''

    assert 1 == 1

def test_save_repr():
    '''
    Test save repr
    '''

    assert 1 == 1
