from itertools import combinations
import pandas as pd
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
    
    pairs_A, pairs_B = (np.empty([fppc,3],dtype="<U128"), np.empty([fppc,3],dtype="<U128"))
    k = 0
    for i,j in combinations(range(contig_frags),2):
        if k==fppc:
            break
        if abs(j-i) >= min_dist_in_step:
            pairs_A[k,:] = [label,i,(i+frag_steps)]
            pairs_B[k,:] = [label,j,(j+frag_steps)]            
            k += 1

    if k < fppc:
        print("Error: cannot make {} unique pairs with genome of {} fragments"
              .format(fppc,contig_frags))
        import ipdb;ipdb.set_trace()
        
    dfs = {'A': pd.DataFrame(pairs_A,columns=["sp","start","end"]),
           'B': pd.DataFrame(pairs_B,columns=["sp","start","end"])}
    
    return pd.concat(dfs.values(),axis=1,keys=dfs.keys())

def make_negative_pairs(n_frags_all, n_examples, frag_steps):
    """
    n_frags_all: nb of fragments per genome
    n_examples: nb of pairs to generate
    frag_steps: nb of steps in a fragment
    
    1) select genome pairs
    2) select random fragments
    """

    pair_idx = np.random.choice(len(n_frags_all),
                                [ 5*n_examples, 2 ])
    
    pair_idx = pair_idx[pair_idx[:,0] != pair_idx[:,1] ][:n_examples,:]

    rd_frags = [ np.array([ np.random.choice(n_frags)
                            for n_frags in n_frags_all.values[pair_idx[:,i]] ])
                          for i in range(2) ]

    pairs = [ pd.DataFrame({"sp": n_frags_all.index[pair_idx[:,i]],
                            "start": rd_frags[i],
                            "end": rd_frags[i] + frag_steps})
              for i in range(2) ]
    
    dfs = {'A': pairs[0],
           'B': pairs[1] }
    
    return pd.concat(dfs.values(),axis=1,keys=dfs.keys())

def make_pairs(contigs,step,frag_len,output=None,n_examples=1e6):
    """
    
    """
    contig_frags = pd.Series({ ctg.id: 1+int((len(ctg.seq)-frag_len)/step)
                               for ctg in contigs})
    frag_pairs_per_ctg = int(n_examples / len(contig_frags) / 2)
    frag_steps = int(frag_len/step)
    
    positive_pairs = pd.concat([ make_positive_pairs(ctg,frag_steps,genome_frags,frag_pairs_per_ctg)
                                 for ctg,genome_frags in progressbar(contig_frags.items(),
                                                                     max_value=len(contigs)) ])
    negative_pairs = make_negative_pairs(contig_frags, int(n_examples/2), frag_steps)

    all_pairs = pd.concat([positive_pairs,negative_pairs]).sample(frac=1)
    all_pairs.index = np.arange(len(all_pairs)).astype(int)

    all_pairs.loc[:,("A",["start","end"])] = all_pairs.loc[:,("A",["start","end"])].astype(int)
    all_pairs.loc[:,("A",["start","end"])] *= step
    all_pairs.loc[:,("B",["start","end"])] = all_pairs.loc[:,("B",["start","end"])].astype(int)    
    all_pairs.loc[:,("B",["start","end"])] *= step    

    if output is not None:
        all_pairs.to_csv(output)

    else:
        return all_pairs
