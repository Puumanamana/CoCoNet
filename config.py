dataset = "sim"

io_path = {
    'in': "input_data/{}".format(dataset),
    'out': "output_data/{}".format(dataset)
}

n_examples = { 'train': int(1e6), 'test': int(5e4) }
        
frag_len = 1024
min_genome_length = 2*frag_len
step = int(frag_len/8)
kmer = 4
model_type = 'CoCoNet'

train_args = {
    'batch_size': 64,
    'learning_rate': 1e-4,
    'window_size': 15,
    'load_batch': 2000,
    'kmer': kmer,
    'rc': False
}

nn_arch = {
    'composition': { 'neurons': [128,64,32] },
    'coverage': { 'neurons': [128,64,32],
                  'n_filters': 64, 'kernel_size': 7,'conv_stride': 2,
                  'pool_size': 3, 'pool_stride': 2},
    'combination': { 'neurons': [64] }
}

cluster_args = {
    'n_frags': 30,
    'max_neighbor': 100,
    'prob_thresh': 0.75,
    'hits_threshold': 0.95    
}

