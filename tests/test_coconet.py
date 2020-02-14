'''
Tests for main algorithm
'''

from pathlib import Path

from coconet.coconet import main
from coconet.parser import parse_args

LOCAL_DIR = Path(__file__).parent
TEST_DIR = "{}/tests/sim_data".format(Path(__file__).parent.parent)

PARAMS = {
    'fasta': Path("{}/assembly.fasta".format(TEST_DIR)),
    'coverage': list(Path(TEST_DIR).glob("sample_*.bam")),
    'output': Path('./output_test'),
    'n_train': 64,
    'n_test': 8,
    'batch_size': 2,
    'min_prevalence': 0,
    'test_ratio': 0.2,
    'threads': 1,
    'n_frags': 5,
    'compo_neurons': [8, 4],
    'cover_neurons': [8, 4],
    'cover_kernel': 2,
    'wsize': 2,
    'wstep': 2
}

def test_parser():
    '''
    Test parser
    '''

    args = parse_args()

    assert isinstance(args.kmer, int)

def test_all():
    '''
    Test overall app
    '''

    fail = True

    try:
        main(**PARAMS)
        fail = False
    except Exception as error:
        print(error)

    for filepath in PARAMS['output'].glob('./*'):
        filepath.unlink()
    PARAMS['output'].rmdir()

    assert not fail

if __name__ == '__main__':
    # test_parser()
    test_all()
