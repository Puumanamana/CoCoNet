from itertools import combinations
import numpy as np

from progressbar import progressbar

def calculate_optimal_dist(n_frags,fppc):
    """
    For a given contig, get the maximum distance between fragments, s.t.:
       - A fragment step (step)
       - A number of fragment pairs per contig (fppc)
       - A number of fragment per contig (n_frags)
    Explanation for formula in paper
    """

    min_dist_in_steps = int(n_frags+0.5*(1-np.sqrt(8*fppc+1)))

    return min_dist_in_steps

def make_positive_pairs(label,frag_steps,contig_frags,fppc):

    min_dist_in_step = calculate_optimal_dist(contig_frags,fppc)
    
    pairs = np.recarray([fppc,2],
                        dtype=[('sp','<U128'),('start','uint32'),('end','uint32')])
    k = 0
    for i,j in combinations(range(contig_frags),2):
        if k==fppc:
            break
        if abs(j-i) >= min_dist_in_step:
            pairs[k] = [(label,i,(i+frag_steps)), (label,j,(j+frag_steps))]
            k += 1

    if k < fppc:
        print("\nWARNING: cannot make {} unique pairs with genome of {} fragments"
              .format(fppc,contig_frags))
        print("Selection random pairs with replacement")

        pairs.sp = np.tile(label, [fppc,2])
        pairs.start = np.random.choice(frag_steps,[fppc,2])
        pairs.end = pairs.start + frag_steps

    return pairs

def make_negative_pairs(n_frags_all, n_examples, frag_steps):
    """
    n_frags_all: nb of fragments per genome
    n_examples: nb of pairs to generate
    frag_steps: nb of steps in a fragment
    
    1) select genome pairs
    2) select random fragments
    """

    pairs = np.recarray([n_examples,2],
                        dtype=[('sp','<U128'),('start','uint32'),('end','uint32')])
    
    pair_idx = np.random.choice(len(n_frags_all),
                                [ 5*n_examples, 2 ])
    
    pair_idx = pair_idx[pair_idx[:,0] != pair_idx[:,1] ][:n_examples,:]

    rd_frags = np.array([[ np.random.choice(n_frags_all[ctg])
                           for ctg in pair_idx[:,i] ]
                         for i in range(2) ]).T

    pairs['sp'] = pair_idx
    pairs['start'] = rd_frags
    pairs['end'] = rd_frags + frag_steps
    
    return pairs

def make_pairs(contigs,step,frag_len,output=None,n_examples=1e6):
    """
    
    """

    contig_frags = np.array([ 1+int((len(ctg.seq)-frag_len)/step)
                              for ctg in contigs ])
    frag_pairs_per_ctg = int(n_examples / len(contig_frags) / 2)
    frag_steps = int(frag_len/step)
    
    positive_pairs = np.vstack([ make_positive_pairs(idx,frag_steps,genome_frags,frag_pairs_per_ctg)
                                 for idx,genome_frags in progressbar(enumerate(contig_frags),
                                                                     max_value=len(contigs)) ])
    negative_pairs = make_negative_pairs(contig_frags, int(n_examples/2), frag_steps)

    all_pairs = np.vstack([positive_pairs,negative_pairs])
    np.random.shuffle(all_pairs)
    
    contig_names = np.array([ ctg.id for ctg in contigs ])
    all_pairs['sp'] = contig_names[all_pairs['sp'].astype(int)]
    all_pairs['start'] *= step
    all_pairs['end'] *= step
    
    np.save(output,all_pairs)

    return all_pairs
