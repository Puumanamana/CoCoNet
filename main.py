'''
Root script to run CoCoNet

Inputs:
Outputs:
'''

import os
import subprocess
from pathlib import Path
import itertools
import argparse
import copy
from math import ceil

import h5py
from Bio import SeqIO
import numpy as np
import torch

from preprocessing import format_assembly, bam_list_to_h5, filter_h5
from fragmentation import make_pairs
from dl_util import initialize_model, train, test_summary
from clustering import save_repr_all, fill_adjacency_matrix, iterate_clustering

def parse_args():
    '''
    Parse arguments to run CoCoNet
    '''

    subparser_descr = {
        'preprocessing': 'Preprocess the contig coverage',
        'fragmentation': 'Fragmentation tool to split the contigs',
        'deepLearning': 'Train neural network',
        'clustering': 'Cluster contigs',
        'run': 'Run CoCoNet from the beginning'
    }

    group_kwargs = {
        'io': [
            {'dest': '--fasta', 'type': str, 'help': 'Path to assembly FASTA file'},
            {'dest': '--coverage', 'type': str, 'nargs': '+', 'help': 'Path to .bam directory or .h5 coverage file'},
            {'dest': '--output', 'type': str, 'default': 'output_data', 'help': 'Path to output directory'}],
        'other arguments': [
            {'dest': '--name', 'type': str, 'default': 'Delong_velvet', 'help': 'Dataset name'},
            {'dest': '--fl', 'type': int, 'default': 1024, 'help': 'Fragment length for contig splitting'},
            {'dest': '--kmer', 'type': int, 'default': 4, 'help': 'k-mer size for composition vector'},            
            {'dest': '--threads', 'type': int, 'default': 10, 'help': 'Number of threads'}],
    }

    parser_kwargs = {
        'preprocessing': [
            {'dest': '--min_ctg_len', 'type': int, 'default': 2048, 'help': 'Minimum contig length'},
            {'dest': '--min_prevalence', 'type': int, 'default': 2, 'help': 'Minimum contig prevalence for binning. Contig with less that value are filtered out.'},
            {'dest': '--min_mapping_quality', 'type': int, 'default': 50, 'help': 'Minimum mapping quality for bam filtering'},
            {'dest': '--flag', 'type': int, 'default': 3596, 'help': 'Sam Flag for bam filtering'},
            {'dest': '--fl-range', 'type': int, 'nargs': 2, 'default': [200, 500], 'help': 'Only allow for paired alignments with spacing within this range'},
            {'dest': '--tmp_dir', 'type': str, 'default': './tmp42', 'help': 'Temporary directory for bam processing'}
        ],
        'fragmentation': [
            {'dest': '--fs', 'type': int, 'default': 128, 'help': 'Fragments spacing'},
            {'dest': '--n_examples_train', 'type': int, 'default': int(1e6), 'help': 'Number of training examples'},
            {'dest': '--n_examples_test', 'type': int, 'default': int(1e4), 'help': 'Number of test examples'}
        ],
        'deepLearning': [
            {'dest': '--batch_size', 'type': int, 'default': 256, 'help': 'Batch size for training'},
            {'dest': '--learning_rate', 'type': float, 'default': 1e-4, 'help': 'Learning rate for gradient descent'},
            {'dest': '--load_batch', 'type': int, 'default': 1000, 'help': 'Number of coverage value to load in memory. Consider lowering this value if your RAM is limited.'},
            {'dest': '--wsize', 'type': int, 'default': 64, 'help': 'Smoothing window size for coverage vector'},
            {'dest': '--wstep', 'type': int, 'default': 32, 'help': 'Subsampling step for coverage vector'},
            {'dest': '--compo_neurons', 'type': int, 'default': [64, 32], 'nargs': 2, 'help': 'Number of neurons for the composition network. Only works with 2 values'},            
            {'dest': '--cover_neurons', 'type': int, 'default': [64, 32], 'nargs': 2, 'help': 'Number of neurons for the coverage network. Only works with 2 values'},
            {'dest': '--cover_filters', 'type': int, 'default': 32, 'help': 'Number of filters for convolution layer of coverage network.'},
            {'dest': '--cover_kernel_size', 'type': int, 'default': 7, 'help': 'Kernel size for convolution layer of coverage network.'},
            {'dest': '--cover_stride', 'type': int, 'default': 3, 'help': 'Convolution stride for convolution layer of coverage network.'},
            {'dest': '--combined_neurons', 'type': int, 'default': 32, 'help': 'Number of neurons for the merging network. Only works with 1 value'},
            {'dest': '--no-rc', 'action': 'store_true', 'default': False, 'help': 'Do not add the reverse complement k-mer occurrences to the composition vector'},
            {'dest': '--norm', 'action': 'store_true', 'default': False, 'help': 'Normalize the k-mer occurrences to frequencies'},
        ],
        'clustering': [
            {'dest': '--n_frags', 'type': int, 'default': 30, 'help': 'Number of fragments to split a contigs'},
            {'dest': '--max_neighbors', 'type': int, 'default': 100, 'help': 'Maximum number of neighbors to consider to compute the adjacency matrix'},
            {'dest': '--hits_threshold', 'type': float, 'default': 0.8, 'help': 'Minimum percent of edges between two contigs to form an edge between them.'},
            {'dest': '--gamma1', 'type': float, 'default': 0.1, 'help': 'CPM optimization value for the first run of the Leiden clustering'},
            {'dest': '--gamma2', 'type': float, 'default': 0.75, 'help': 'CPM optimization value for the second run of the Leiden clustering'}
        ]
    }

    parser_kwargs['run'] = list(itertools.chain.from_iterable(
        copy.deepcopy(parser_kwargs).values()
    ))

    parser = argparse.ArgumentParser(prog='CoCoNet')

    for name, entries in group_kwargs.items():
        group = parser.add_argument_group(name)
        for entry in entries:
            dest = entry.pop('dest')
            group.add_argument(dest, **entry)

    subparsers = parser.add_subparsers(help='commands', dest='command')

    for sub_cmd, entries in parser_kwargs.items():
        subparser = subparsers.add_parser(sub_cmd, help=subparser_descr[sub_cmd])

        for entry in entries:
            dest = entry.pop('dest')
            subparser.add_argument(dest, **entry)

    args = parser.parse_args()

    return args

def check_inputs(args):
    '''
    Check if all input files exist and have the right extension
    '''

    args.fasta = Path(args.fasta)

    if not args.fasta.is_file():
        print("{} is not a file. Please correct")
        exit(42)

    args.coverage = [Path(cov) for cov in args.coverage]

    if (len(args.coverage) == 1) and (args.coverage[0].is_file()):
        suffixes = args.coverage[0].suffixes
        ext = suffixes.pop()

        if ext == '.gz':
            subprocess.check_output(['gunzip', args.coverage[0]])
            args.coverage[0] = args.coverage[0].replace('{}$'.format(ext, ''))
            ext = suffixes.pop()

        if ext == '.h5':
            args.coverage = args.coverage[0]

    else:
        for bam in args.coverage:
            if not bam.is_file() or not bam.suffix == '.bam':
                print('{} is nto a valid file or is not bam formatted'.format(bam))
                exit(42)

def set_outputs(args):

    args.output = Path(args.output)
    args.output.mkdir(exist_ok=True)

    args.filt_fasta = Path('{}/filtered_{}'.format(args.output, args.fasta.name))
    args.filt_h5 = Path('{}/filtered_{}'.format(args.output, args.h5.name))
    args.singleton_file = Path('{}/filtered_{}'.format(args.output, args.h5.name))
    args.pairs = {'test': Path('{}/pairs_test.npy'.format(args.output)),
                  'train': Path('{}/pairs_train.npy'.format(args.output))}
    args.model = Path('{}/CoCoNet.pth'.format(args.output))
    args.nn_test = Path('{}/CoCoNet_test.csv'.format(args.output))
    args.repr = {'composition': Path('{}/representation_compo.h5'.format(args.output)),
                 'coverage': Path('{}/representation_cover.h5'.format(args.output))}

    if hasattr(args, 'hits_threshold'):
        args.adjacency_matrix = Path('{}/adjacency_matrix_nf{}.npy'
                                     .format(args.output, args.n_frags))
        args.refined_adjacency_matrix = Path('{}/adjacency_matrix_nf{}_refined.npy'
                                             .format(args.output, args.n_frags))
        args.assignments = Path(
            '{}/leiden-{}-{}_nf{}.csv'.format(args.output, args.hits_threshold,
                                              args.gamma1, args.n_frags)
        )
        args.refined_assignments = Path(
            '{}/leiden-{}-{}-{}_nf{}.csv'.format(args.output, args.hits_threshold,
                                                 args.gamma1, args.gamma2, args.n_frags))

def get_input_shapes(kmer, fragment_length, rev_compl, h5, wsize, wstep):
    with h5py.File(h5, 'r') as handle:
        n_samples = handle.get(list(handle.keys())[0]).shape[0]

    input_shapes = {
        'composition': 4**kmer * (1-rev_compl) + 136 * rev_compl, # Fix the 136 with the good calculation
        'coverage': (ceil((fragment_length-wsize+1) / wstep), n_samples)
    }

    return input_shapes

def get_architecture(args):
    return {
        'composition': {'neurons': args.compo_neurons},
        'coverage': {'neurons': args.cover_neurons,
                     'n_filters': args.cover_filters,
                     'kernel_size': args.cover_kernel_size,
                     'conv_stride': args.conv_stride},
        'combined': {'neurons': args.combined_neurons}
    }

def run_preprocessing(args):

    # Input is bam formatted
    if isinstance(args.coverage, list):
        if not args.filt_fasta.is_file():
            format_assembly(args.fasta, args.filt_fasta, min_length=args.min_ctg_len)
        if not args.filt_h5.is_file():
            bam_list_to_h5(
                fasta=args.filt_fasta, coverage_bam=args.coverage, output=args.filt_h5,
                temp_dir=args.temp_dir, singleton_file=args.singleton_file,
                threads=args.threads,
                min_prevalence=args.min_prevalence,
                min_qual=args.min_mapping_quality,
                flag=args.flag,
                fl_range=args.fl_range
            )

    # Input is h5 formatted
    if not (args.filt_fasta.is_file() and args.filt_h5.is_file()):
        filter_h5(args.fasta, args.coverage,
                  args.filt_fasta, args.filt_h5,
                  min_length=args.min_ctg_len,
                  min_prevalence=args.min_prevalence,
                  singleton_file=args.singleton_file)

def run_fragmentation(args):
    if args.pairs['train'].is_file():
        assembly = [contig for contig in SeqIO.parse(args.filt_fasta, "fasta")]

        assembly_idx = {'test': np.random.choice(len(assembly), int(0.1*len(assembly)))}
        assembly_idx['train'] = np.setdiff1d(range(len(assembly)), assembly_idx['test'])

        n_examples = {'train': args.n_examples_train, 'test': args.n_examples_test}

        for mode, pair_file in args.pairs.items():
            print("Making {} pairs".format(mode))
            make_pairs([assembly[idx] for idx in assembly_idx[mode]],
                       args.fs, args.fl, pair_file, n_examples=n_examples[mode])

def run_deep_learning(args):
    torch.set_num_threads(args.threads)

    input_shapes = get_input_shapes(args.kmer, args.fl, args.rc,
                                    args.filt_h5, args.wsize, args.wstep)
    architecture = get_architecture(args)

    model = initialize_model('CoCoNet', input_shapes, architecture)

    if not args.model.is_file():
        train(model, args)

    checkpoint = torch.load(args.model)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    if not os.path.exists(args.repr['coverage']):
        save_repr_all(model, args)

    return model

def run_clustering(args):

    checkpoint = torch.load(args.model)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    _ = test_summary(model, args=args)
    
    fill_adjacency_matrix(model, args)

    if not args.refined_assignments.is_file():
        iterate_clustering(model, args)

def run():
    '''
    CoCoNet runner
    '''

    args = parse_args()
    check_inputs(args)
    set_outputs(args)

    if args.command == 'preprocessing':
        run_preprocessing(args)
    elif args.command == 'fragmentation':
        run_fragmentation(args)
    elif args.command == 'deepLearning':
        run_deep_learning(args)
    elif args.command == 'clustering':
        run_clustering(args)
    elif args.command == 'run':
        run_preprocessing(args)
        run_fragmentation(args)
        run_deep_learning(args)
        run_clustering(args)
    else:
        print('Unknown command {}'.format(args.command))
        exit(42)
        
if __name__ == '__main__':
    run()
