dataset = "sim"

io_path = {
    'in': "input_data/{}".format(dataset),
    'out': "output_data/{}".format(dataset)
}

frag_len = 1024
min_genome_length = 2*frag_len
step = int(frag_len/8)
n_frags = 50

