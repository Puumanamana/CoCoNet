dataset = "sim"

io_path = {
    'in': "input_data/{}".format(dataset),
    'out': "output_data/{}".format(dataset)
}

frag_len = 1024
min_genome_length = 2*frag_len
step = int(frag_len/8)
kmer = 4
model_type = 'CoCoNet'

train_args = {
    'batch_size': 64,
    'learning_rate': 1e-4,
    'window_size': 16,
    'load_batch': 200,
    'kmer': kmer,
    'rc': False
}

nn_arch = {
    'composition': { 'neurons': [128,64,32] },
    'coverage': { 'neurons': [128,64,32],
                  'n_filters': 64, 'kernel_size': 16,'conv_stride': 8,
                  'pool_size': 4, 'pool_stride': 2},
    'combination': { 'neurons': [64] }
}

cluster_args = {
    'n_frags': 20,
    'max_neighbor': 50,
    'prob_thresh': 0.75,
    'hits_threshold': 0.95    
}

