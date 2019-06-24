from config import io_path
from config import frag_len,step,n_frags
from fragmentation import make_pairs

def run():

    # format_assembly()
    
    assembly = [ contig for contig in
                 SeqIO.parse("{}/assembly.fasta".format(io_path["in"]),"fasta") ]
    
    assembly_test_idx = np.random.choice(len(assembly),int(0.05*len(assembly)))
    assembly_train_idx = np.setdiff1d(range(len(assembly)),
                                      assembly_test_idx)
    print("Making test pairs")
    make_pairs([ assembly[idx] for idx in assembly_test_idx ],
               n_frags,step,frag_len,
               "{}/pairs_test.csv".format(io_path["out"]),
               N_samples=1e4)

    print("Making training pairs")    
    make_pairs([ assembly[idx] for idx in assembly_train_idx ],
               n_frags,step,frag_len,
               "{}/pairs_train.csv".format(io_path["out"]),
               N_samples=1e6)

if __name__ == '__main__':
    run()

