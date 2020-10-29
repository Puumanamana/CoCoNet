"""
Tools to split contigs into smaller fragment
to train the neural network
"""

import logging
from math import ceil
from itertools import combinations
import numpy as np

from coconet.tools import run_if_not_exists


logger = logging.getLogger('<learning>')

def calculate_optimal_dist(n_frags, fppc):
    """
    For a given contig, get the maximum distance between fragments/
    Explanation for formula in paper

    Args:
        n_frags (int): Number of fragments
        fppc (int): Number of fragments pairs per contig
    Returns:
        int: maximum distance between fragments
    """

    min_dist_in_steps = int(n_frags+0.5*(1-np.sqrt(8*fppc+1)))

    return min_dist_in_steps

def make_positive_pairs(label, frag_steps, contig_frags, fppc, encoding_len=128):
    """
    Select fragments as distant as possible for the given contig

    Args:
        label (int): contig name
        frag_steps (int): number of steps in a fragment
        contig_frags (int): Number of fragments in contig
        fppc (int): Number of fragments to generate
        encoding_len (int): contig name encoding length (to save space)
    Returns:
        np.array: fragment pairs for contig label
    """

    min_dist_in_step = calculate_optimal_dist(contig_frags, fppc)

    pairs = np.zeros(
        (fppc, 2),
        dtype=[('sp', f'<U{encoding_len}'), ('start', 'uint32'), ('end', 'uint32')]
    )

    k = 0
    for i, j in combinations(range(contig_frags), 2):
        if k == fppc:
            break
        if abs(j-i) >= min_dist_in_step:
            pairs[k] = [(label, i, (i+frag_steps)), (label, j, (j+frag_steps))]
            k += 1

    if k < fppc:
        pairs['sp'] = np.tile(label, [fppc, 2])
        pairs['start'] = np.random.choice(contig_frags, [fppc, 2])
        pairs['end'] = pairs['start'] + frag_steps

    return pairs

def make_negative_pairs(n_frags_all, n_examples, frag_steps, encoding_len=128):
    """
    1) select genome pairs
    2) select random fragments

    Args:
        n_frags_all (int): nb of fragments per genome
        n_examples (int): nb of pairs to generate
        frag_steps (int): number of steps in a fragment
        encoding_len (int): contig name encoding length (to save space)
    Returns:
        np.array: fragment pairs for each distinct contig pair
    """

    pairs = np.zeros(
        [n_examples, 2],
        dtype=[('sp', f'<U{encoding_len}'), ('start', 'u4'), ('end', 'u4')]
    )

    pair_idx = np.random.choice(len(n_frags_all),
                                [5*n_examples, 2])

    cond = pair_idx[:, 0] != pair_idx[:, 1]
    pair_idx = np.unique(pair_idx[cond], axis=0)[:n_examples, :]

    if pair_idx.size == 0:
        logger.fatal('No contigs found in data. Aborting')
        raise RuntimeError

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

    Args:
        contigs (list): (name, sequence) of a set of contigs
        step (int): distance between two consecutive fragments
        frag_len (int): length of contig substrings
        output (str): path to save pairs
        n_examples (int): number of examples to generate

    Returns:
        np.recarray: fragment pairs for each pair
    """

    contig_frags = np.array([(1+len(ctg)-frag_len)//step
                             for (name, ctg) in contigs])

    max_encoding = np.max([len(name) for (name, _) in contigs])

    pairs_per_ctg = ceil(n_examples / 2 / len(contig_frags))
    frag_steps = frag_len // step

    positive_pairs = np.vstack([
        make_positive_pairs(idx, frag_steps, genome_frags, pairs_per_ctg, encoding_len=max_encoding)
        for idx, genome_frags in enumerate(contig_frags)
    ])

    negative_pairs = make_negative_pairs(contig_frags, len(positive_pairs),
                                         frag_steps, encoding_len=max_encoding)

    all_pairs = np.vstack([positive_pairs[:n_examples//2],
                           negative_pairs[:n_examples//2]])

    np.random.shuffle(all_pairs)

    contig_names = np.array([name for (name, ctg) in contigs])
    all_pairs['sp'] = contig_names[all_pairs['sp'].astype(int)]
    all_pairs['start'] *= step
    all_pairs['end'] *= step

    all_pairs = all_pairs.view(np.recarray)

    if output is not None:
        np.save(output, all_pairs)

    return all_pairs
