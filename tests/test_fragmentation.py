import unittest

import os
import sys
sys.path.append(os.path.expanduser("~/CoCoNet"))
from itertools import combinations

from fragmentation import calculate_optimal_dist

def get_pairs_frag_with_dist(n_frags,dist):
    count = 0
    for i,j in combinations(range(n_frags),2):
        if abs(i-j) >= dist:
            count += 1
    return count
    

class TestFragmentation(unittest.TestCase):
    """ """

    def test_optimal_dist(self, n_frags=50, fppc=5):
        
        dist = calculate_optimal_dist(n_frags,fppc)
        
        total_pairs_opt = get_pairs_frag_with_dist(n_frags,dist)
        total_pairs_more = get_pairs_frag_with_dist(n_frags,dist+1)

        is_opt = ( (total_pairs_more < fppc) # With a bigger distance, we have less than fppc fragments
                   & (total_pairs_opt >= fppc) )

        print("{} < {} < {}".format(total_pairs_more, fppc, total_pairs_opt))

        assert is_opt

if __name__ == "__main__":
    unittest.main()
