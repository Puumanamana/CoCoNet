dataset = "giant_viruses"

io_path = {
    'in': "input_data/{}".format(dataset),
    'out': "output_data/{}".format(dataset)
}

n_examples = { 'train': int(2e6), 'test': int(5e4) }

frag_len = 1024
min_contig_length = 2*frag_len
step = int(frag_len/16)
kmer_list = [2,3,4]
model_type = 'CoCoNet'

train_args = {
    'batch_size': 64,
    'learning_rate': 1e-4,
    'window_size': 15,
    'load_batch': 2000,
    'kmer_list': kmer_list,
    'rc': False
}

nn_arch = {
    'composition': { 'neurons': [128,64,32] },
    'coverage': { 'neurons': [128,64,32],
                  'n_filters': 64, 'kernel_size': 7,'conv_stride': 3,
                  'pool_size': 2, 'pool_stride': 1},
    'combination': { 'neurons': [32] }
}

cluster_args = {
    'n_frags': 30,
    'max_neighbor': 50,
    'prob_thresh': 0.8,
    'hits_threshold': 0.95
}

