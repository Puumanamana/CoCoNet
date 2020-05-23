'''
Tools to split contigs into smaller fragment
to train the neural network
'''

import sys
from math import ceil
from itertools import combinations
import numpy as np

from coconet.tools import run_if_not_exists

def vstack_recarrays(arrays):
    '''
    numpy.vstack destructures the recarray.
    Solution taken from Stackoverflow:
    stackoverflow.com/questions/1791791/stacking-numpy-recarrays-without-losing-their-recarrayness/14613280
    '''

    return arrays[0].__array_wrap__(np.vstack(arrays))

def calculate_optimal_dist(n_frags, fppc):
    """
    For a given contig, get the maximum distance between fragments, s.t.:
       - A fragment step (step)
       - A number of fragment pairs per contig (fppc)
       - A number of fragment per contig (n_frags)
    Explanation for formula in paper
    """

    min_dist_in_steps = int(n_frags+0.5*(1-np.sqrt(8*fppc+1)))

    return min_dist_in_steps

def make_positive_pairs(label, frag_steps, contig_frags, fppc, encoding_len=128):
    '''
    Select fragments as distant as possible for the given contig
    '''

    min_dist_in_step = calculate_optimal_dist(contig_frags, fppc)

    pairs = np.recarray([fppc, 2],
                        dtype=[('sp', '<U{}'.format(encoding_len)),
                               ('start', 'uint32'),
                               ('end', 'uint32')])
    k = 0
    for i, j in combinations(range(contig_frags), 2):
        if k == fppc:
            break
        if abs(j-i) >= min_dist_in_step:
            pairs[k] = [(label, i, (i+frag_steps)), (label, j, (j+frag_steps))]
            k += 1

    if k < fppc:
        # print("WARNING: cannot make {} unique pairs with genome of {} fragments".format(fppc, contig_frags), end='\r')
        pairs.sp = np.tile(label, [fppc, 2])
        pairs.start = np.random.choice(contig_frags, [fppc, 2])
        pairs.end = pairs.start + frag_steps

    return pairs

def make_negative_pairs(n_frags_all, n_examples, frag_steps, encoding_len=128):
    """
    n_frags_all: nb of fragments per genome
    n_examples: nb of pairs to generate
    frag_steps: nb of steps in a fragment
    1) select genome pairs
    2) select random fragments
    """

    pairs = np.recarray([n_examples, 2],
                        dtype=[('sp', '<U{}'.format(encoding_len)),
                               ('start', 'uint32'),
                               ('end', 'uint32')])

    pair_idx = np.random.choice(len(n_frags_all),
                                [5*n_examples, 2])

    cond = pair_idx[:, 0] != pair_idx[:, 1]
    pair_idx = np.unique(pair_idx[cond], axis=0)[:n_examples, :]

    if pair_idx.size == 0:
        sys.exit("No contigs found in data. Aborting")

    if len(pair_idx) < n_examples:
        pair_idx = np.vstack(np.triu_indices(len(n_frags_all), k=1)).T
        subset = np.random.choice(len(pair_idx), n_examples)
        pair_idx = pair_idx[subset]

    rd_frags = np.array([[np.random.choice(n_frags_all[ctg])
                          for ctg in pair_idx[:, i]]
                         for i in range(2)]).T

    pairs['sp'] = pair_idx
    pairs['start'] = rd_frags
    pairs['end'] = rd_frags + frag_steps

    return pairs

@run_if_not_exists()
def make_pairs(contigs, step, frag_len, output=None, n_examples=1e6):
    """
    Extract positive and negative pairs for [contigs]
    """

    contig_frags = np.array([(1+len(ctg.seq)-frag_len)//step
                             for ctg in contigs])

    max_encoding = np.max([len(ctg.id) for ctg in contigs])

    pairs_per_ctg = ceil(n_examples / 2 / len(contig_frags))
    frag_steps = frag_len // step

    positive_pairs = vstack_recarrays([
        make_positive_pairs(idx, frag_steps, genome_frags, pairs_per_ctg, encoding_len=max_encoding)
        for idx, genome_frags in enumerate(contig_frags)
    ])

    negative_pairs = make_negative_pairs(contig_frags, len(positive_pairs), frag_steps,
                                         encoding_len=max_encoding)

    all_pairs = vstack_recarrays([positive_pairs, negative_pairs])

    np.random.shuffle(all_pairs)

    contig_names = np.array([ctg.id for ctg in contigs])
    all_pairs['sp'] = contig_names[all_pairs['sp'].astype(int)]
    all_pairs['start'] *= step
    all_pairs['end'] *= step

    if output is not None:
        np.save(output, all_pairs)

    return all_pairs
