'''
Unittest for train / test examples generation
'''

from itertools import combinations

import numpy as np

from coconet.fragmentation import calculate_optimal_dist
from coconet.fragmentation import make_negative_pairs, make_positive_pairs, make_pairs

from .data import generate_fasta_file

def get_pairs_frag_with_dist(n_frags, dist):
    '''
    Ground truth for calculation of all pairs within distance >= [dist]
    '''

    count = 0
    for i, j in combinations(range(n_frags), 2):
        if abs(i-j) >= dist:
            count += 1
    return count

def test_optimal_dist(n_frags=50, fppc=5):
    '''
    Test if math formula for optimal dist works
    '''

    dist = calculate_optimal_dist(n_frags, fppc)

    total_pairs_opt = get_pairs_frag_with_dist(n_frags, dist)
    total_pairs_more = get_pairs_frag_with_dist(n_frags, dist+1)

    # With a bigger distance, we have less than fppc fragments
    is_opt = ((total_pairs_more < fppc)
              & (total_pairs_opt >= fppc))

    print("{} < {} < {}".format(total_pairs_more, fppc, total_pairs_opt))

    assert is_opt

def test_negative_pairs(n_examples=10):
    '''
    Test negative examples (pairs from different genomes)
    '''

    fragments = np.array([200, 100, 100, 90, 200])
    result = make_negative_pairs(fragments, n_examples, 3)

    assert sum(result['sp'][:, 0] != result['sp'][:, 1]) == n_examples
    assert np.array(result.tolist()).shape == (n_examples, 2, 3)

def test_negative_pairs_with_few_ctg(n_examples=100):
    '''
    Test negative examples with too many examples
    '''

    fragments = np.array([200, 100, 100, 90])
    result = make_negative_pairs(fragments, n_examples, 3)

    assert sum(result['sp'][:, 0] != result['sp'][:, 1]) == n_examples
    assert np.array(result.tolist()).shape == (n_examples, 2, 3)

def test_positive_pairs(fppc=30, contig_frags=100):
    '''
    Test positive examples (pairs from the same genome)
    '''

    min_dist = calculate_optimal_dist(contig_frags, fppc)
    result = make_positive_pairs(0, 4, contig_frags, fppc, encoding_len=8)

    assert sum(result['sp'][:, 0] == result['sp'][:, 1]) == fppc
    assert sum(np.abs(result['start'][:, 0].astype(int)
                      - result['start'][:, 1].astype(int)) < min_dist) == 0
    assert np.array(result.tolist()).shape == (fppc, 2, 3)

def test_all_pairs(n_examples=50):
    '''
    Test wrapper to get both positive and negative examples in equal amounts
    '''

    contigs = generate_fasta_file(10, 20, save=False)
    contigs = [(ctg.id, str(ctg.seq)) for ctg in contigs]

    step = 2
    frag_len = 3

    result = make_pairs(contigs, step, frag_len, n_examples=n_examples)

    assert 0.4 < np.mean(result['sp'][:, 0] == result['sp'][:, 1]) < 0.6
    assert np.array(result.tolist()).shape[1:] == (2, 3)
    assert len(result) >= n_examples
