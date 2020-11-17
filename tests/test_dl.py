
'''
Tests for deep learning procedure
'''

from pathlib import Path

import torch
from Bio import SeqIO
import numpy as np
import h5py

from coconet.core.config import Configuration
from coconet.fragmentation import make_pairs
from coconet.dl_util import initialize_model, load_model, train, save_repr_all
from coconet.dl_util import get_npy_lines, get_labels, get_test_scores
from coconet.core.torch_models import CompositionModel, CoverageModel, CoCoNet
from coconet.core.generators import CompositionGenerator, CoverageGenerator

from .data import generate_fasta_file, generate_h5_file, generate_pair_file
from .data import FL, STEP, WSIZE, WSTEP
from .data import TEST_LEARN_PRMS, TEST_CTG_LENGTHS, TEST_ARCHITECTURE, TEST_SHAPES

LOCAL_DIR = Path(__file__).parent

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

    fasta_file = generate_fasta_file(*TEST_CTG_LENGTHS, filename='coverage.h5')
    contigs = [(ctg.id, str(ctg.seq)) for ctg in SeqIO.parse(fasta_file, 'fasta')]
    pairs_file = Path('pairs.npy').resolve()

    make_pairs(contigs, STEP, FL, output=pairs_file, n_examples=50)

    gen = CompositionGenerator(pairs_file, fasta_file,
                               batch_size=TEST_LEARN_PRMS['batch_size'],
                               kmer=TEST_LEARN_PRMS['kmer'],
                               rc=TEST_LEARN_PRMS['rc'])

    X1, X2 = next(gen)

    fasta_file.unlink()
    pairs_file.unlink()

    assert X1.shape == X2.shape
    assert X1.shape == (TEST_LEARN_PRMS['batch_size'], 136)

def test_load_data_cover():
    '''
    Test coverage generator
    '''

    contigs = generate_fasta_file(*TEST_CTG_LENGTHS, save=False)
    coverage_file = generate_h5_file(*TEST_CTG_LENGTHS, filename='coverage.h5')
    pairs_file = Path('pairs.npy').resolve()

    contigs = [(seq.id, str(seq.seq)) for seq in contigs]
    make_pairs(contigs, STEP, FL, output=pairs_file, n_examples=50)

    gen = CoverageGenerator(pairs_file, coverage_file,
                            batch_size=TEST_LEARN_PRMS['batch_size'],
                            load_batch=TEST_LEARN_PRMS['load_batch'],
                            wsize=TEST_LEARN_PRMS['wsize'],
                            wstep=TEST_LEARN_PRMS['wstep'])

    X1, X2 = next(gen)

    pairs_file.unlink()
    coverage_file.unlink()

    assert X1.shape == X2.shape
    assert X1.shape == (TEST_LEARN_PRMS['batch_size'], 2, 9)

def test_composition_model():
    '''
    Test if composition model can compute an output
    '''

    model = initialize_model('composition', TEST_SHAPES['composition'], TEST_ARCHITECTURE['composition'])
    x_rd = [torch.FloatTensor(4, TEST_SHAPES['composition']).random_(0, 10) for _ in range(2)]
    y_rd = torch.FloatTensor(4, 1).random_(0, 1)

    pred = model(*x_rd)
    loss = model.compute_loss(pred, y_rd)

    assert 'composition' in pred
    assert y_rd.shape == pred['composition'].shape
    assert isinstance(loss, torch.FloatTensor)

def test_coverage_model():
    '''
    Test if coverage model can compute an output
    '''
    model = initialize_model('coverage', TEST_SHAPES['coverage'], TEST_ARCHITECTURE['coverage'])
    x_rd = [torch.FloatTensor(4, *TEST_SHAPES['coverage'][::-1]).random_(0, 10) for _ in range(2)]
    y_rd = torch.FloatTensor(4, 1).random_(0, 1)

    pred = model(*x_rd)
    loss = model.compute_loss(pred, y_rd)

    assert 'coverage' in pred
    assert y_rd.shape == pred['coverage'].shape
    assert isinstance(loss, torch.FloatTensor)

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

    coverage_file = generate_h5_file(*TEST_CTG_LENGTHS, filename='coverage.h5')
    fasta_file = generate_fasta_file(*TEST_CTG_LENGTHS, save=True)

    fasta = [(seq.id, str(seq.seq)) for seq in SeqIO.parse(fasta_file, 'fasta')]

    make_pairs(fasta, STEP, FL, output=pair_files['train'], n_examples=50)
    make_pairs(fasta, STEP, FL, output=pair_files['test'], n_examples=5)

    train(model, fasta_file, coverage_file, pair_files, results_file, output=model_file,
          **TEST_LEARN_PRMS)

    tests = model_file.is_file() and results_file.is_file()

    for path in list(pair_files.values()) + [fasta_file, coverage_file, model_file, results_file]:
        path.unlink()

    assert tests

def test_load_model():
    '''
    Test if model can be loaded
    '''

    args = {'compo_neurons': TEST_ARCHITECTURE['composition']['neurons'],
            'cover_neurons': TEST_ARCHITECTURE['coverage']['neurons'],
            'cover_filters': TEST_ARCHITECTURE['coverage']['n_filters'],
            'cover_kernel': TEST_ARCHITECTURE['coverage']['kernel_size'],
            'cover_stride': TEST_ARCHITECTURE['coverage']['conv_stride'],
            'merge_neurons': TEST_ARCHITECTURE['merge']['neurons'],
            'kmer': 4, 'no_rc': True,
            'fragment_length': FL, 'wsize': WSIZE, 'wstep': WSTEP}

    cfg = Configuration()
    cfg.init_config(output='.', **args)
    cfg.io['h5'] = generate_h5_file(FL, filename='coverage.h5')

    model = initialize_model('CoCoNet', cfg.get_input_shapes(), cfg.get_architecture())
    model_path = Path('CoCoNet.pth')

    torch.save({
        'state': model.state_dict()
    }, model_path)

    loaded_model = load_model(cfg)

    model_path.unlink()
    cfg.io['h5'].unlink()

    assert isinstance(loaded_model, CoCoNet)

def test_save_repr():
    '''
    Test save repr
    '''

    model = initialize_model('CoCoNet', TEST_SHAPES, TEST_ARCHITECTURE)
    fasta = generate_fasta_file(*TEST_CTG_LENGTHS)
    coverage = generate_h5_file(*TEST_CTG_LENGTHS, filename='coverage.h5')

    output = {k: Path('repr_{}.h5'.format(k))
              for k in ['composition', 'coverage']}

    save_repr_all(model, fasta, coverage, n_frags=5, frag_len=FL, output=output,
                  min_ctg_len=0, wsize=WSIZE, wstep=WSTEP)

    assert all(out.is_file() for out in output.values())

    handles = {k: h5py.File(v, 'r') for (k, v) in output.items()}
    firsts = {k: handle.get(list(handle.keys())[0]).shape
              for k, handle in handles.items()}

    latent_dim = (TEST_ARCHITECTURE['composition']['neurons'][-1]
                  + TEST_ARCHITECTURE['coverage']['neurons'][-1])

    assert firsts['composition'] == (5, latent_dim)
    assert firsts['coverage'] == (5, latent_dim)

    fasta.unlink()
    coverage.unlink()
    for key, filename in output.items():
        handles[key].close()
        filename.unlink()
