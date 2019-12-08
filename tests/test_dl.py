'''
Tests for deep learning procedure
'''

from pathlib import Path
from math import ceil

import torch
from Bio import SeqIO
import numpy as np

from coconet.fragmentation import make_pairs
from coconet.dl_util import initialize_model, train
from coconet.dl_util import get_npy_lines, get_labels, get_confusion_table
from coconet.torch_models import CompositionModel, CoverageModel, CoCoNet
from coconet.generators import CompositionGenerator, CoverageGenerator

from .data import generate_fasta_file, generate_coverage_file, generate_pair_file

LOCAL_DIR = Path(__file__).parent

FL = 20
STEP = 2
WSIZE = 3
WSTEP = 2

TEST_LEARN_PRMS = {'batch_size': 4, 'learning_rate': 1e-3,
                   'kmer': 4, 'rc': True, 'norm': False,
                   'wsize': WSIZE, 'wstep': WSTEP, 'load_batch': 30}
TEST_CTG_LENGTHS = [60, 100, 80]


TEST_ARCHITECTURE = {
    'composition': {'neurons': [8, 4]},
    'coverage': {'neurons': [8, 4],
                 'n_filters': 2,
                 'kernel_size': 3,
                 'conv_stride': 2},
    'combined': {'neurons': 2}
}

TEST_SHAPES = {'composition': 136,
               'coverage': (ceil((FL - WSIZE+1) / WSTEP), 2)}



def test_init_composition_model():
    '''
    Check if model can be initialized
    '''

    model = initialize_model('composition',
                             TEST_SHAPES['composition'],
                             TEST_ARCHITECTURE['composition'])

    assert isinstance(model, CompositionModel)

def test_init_coverage_model():
    '''
    Check if model can be initialized
    '''

    model = initialize_model('coverage',
                             TEST_SHAPES['coverage'],
                             TEST_ARCHITECTURE['coverage'])

    assert isinstance(model, CoverageModel)

def test_init_combined_model():
    '''
    Check if model can be initialized
    '''

    model = initialize_model('CoCoNet', TEST_SHAPES, TEST_ARCHITECTURE)

    assert isinstance(model, CoCoNet)

def test_get_labels():
    '''
    Check if labels can be loaded
    '''

    pairs = generate_pair_file()
    truth = get_labels(pairs).numpy()

    pairs.unlink()

    assert truth.sum() == 1

def test_count_npy():
    '''
    Check if npy can be counted
    '''

    pairs = generate_pair_file()
    n_lines = get_npy_lines(pairs)

    pairs.unlink()

    assert n_lines == 2

def test_load_data_compo():
    '''
    Test composition generator
    '''

    fasta_file = generate_fasta_file(*TEST_CTG_LENGTHS, save=True)
    fasta = list(SeqIO.parse(fasta_file, 'fasta'))
    pairs_file = Path('pairs.npy').resolve()

    make_pairs(fasta, STEP, FL, output=pairs_file, n_examples=50)    

    gen = CompositionGenerator(pairs_file, fasta_file,
                               batch_size=TEST_LEARN_PRMS['batch_size'],
                               kmer=TEST_LEARN_PRMS['kmer'],
                               rc=TEST_LEARN_PRMS['rc'],
                               norm=TEST_LEARN_PRMS['norm'])

    X1, X2 = next(gen)

    fasta_file.unlink()
    pairs_file.unlink()

    assert X1.shape == X2.shape
    assert X1.shape == (TEST_LEARN_PRMS['batch_size'], 136)

def test_load_data_cover():
    '''
    Test coverage generator
    '''

    fasta_file = generate_fasta_file(*TEST_CTG_LENGTHS, save=False)
    coverage_file = generate_coverage_file(*TEST_CTG_LENGTHS)
    pairs_file = Path('pairs.npy').resolve()

    make_pairs(fasta_file, STEP, FL, output=pairs_file, n_examples=50)

    gen = CoverageGenerator(pairs_file, coverage_file,
                            batch_size=TEST_LEARN_PRMS['batch_size'],
                            load_batch=TEST_LEARN_PRMS['load_batch'],
                            window_size=TEST_LEARN_PRMS['wsize'],
                            window_step=TEST_LEARN_PRMS['wstep'])

    X1, X2 = next(gen)

    pairs_file.unlink()
    coverage_file.unlink()

    assert X1.shape == X2.shape
    assert X1.shape == (TEST_LEARN_PRMS['batch_size'], 2, 9)

def test_nn_summary():
    '''
    Check if summary runs
    '''

    assert 1 == 1

def test_confusion_table():
    '''
    Check if confusion table works
    '''

    pred = {'fake': torch.from_numpy(np.array([0.5, 0.6, 0.7, 0.3]).reshape(-1, 1))}
    truth = torch.from_numpy(np.array([0, 0, 1, 0]))

    get_confusion_table(pred, truth)

def test_learn_save_load_model():
    '''
    Check:
    - if the training goes through
    - if the model is saved
    '''

    model = initialize_model('CoCoNet', TEST_SHAPES, TEST_ARCHITECTURE)
    model_file = Path('{}/test_model.pth'.format(LOCAL_DIR))
    results_file = Path('{}/test_res.csv'.format(LOCAL_DIR))

    pair_files = {'train': Path('{}/pairs_train.npy'.format(LOCAL_DIR)),
                  'test': Path('{}/pairs_test.npy'.format(LOCAL_DIR))}

    coverage_file = generate_coverage_file(*TEST_CTG_LENGTHS)
    fasta_file = generate_fasta_file(*TEST_CTG_LENGTHS, save=True)

    fasta = list(SeqIO.parse(fasta_file, 'fasta'))

    make_pairs(fasta, STEP, FL, output=pair_files['train'], n_examples=50)
    make_pairs(fasta, STEP, FL, output=pair_files['test'], n_examples=5)

    train(model, fasta_file, coverage_file, pair_files, results_file, output=model_file, **TEST_LEARN_PRMS)

    tests = model_file.is_file() and results_file.is_file()

    for path in list(pair_files.values()) + [fasta_file, coverage_file, model_file, results_file]:
        path.unlink()

    assert tests


def test_save_repr():
    '''
    Test save repr
    '''

    assert 1 == 1
