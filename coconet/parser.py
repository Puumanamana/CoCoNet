'''
CoCoNet parser information and documentation
'''

import warnings
from pathlib import Path
import argparse

class ToPathAction(argparse.Action):
    '''
    argparse action to convert string to Path objects
    '''
    def __init__(self, option_strings, dest, required=False, **kwargs):

        argparse.Action.__init__(self,
                                 option_strings=option_strings,
                                 dest=dest,
                                 **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        if isinstance(values, (list, tuple)):
            values_ok = [Path(val) for val in values]
        else:
            values_ok = Path(values)

        setattr(namespace, self.dest, values_ok)

def parse_args():
    '''
    '''

    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('-n', '--name', type=str, default='ds', help='Dataset name')
    parser.add_argument('-fl', '--fragment-length', type=int, default=1024, help='Dataset name')
    parser.add_argument('-t', '--threads', type=int, default=20, help='Number of threads')

    io_group = parser.add_argument_group(title='io')
    io_group.add_argument('--fasta', type=str, action=ToPathAction, required=True, help='Path to your assembly file (fasta formatted)')
    io_group.add_argument('--coverage', type=str, nargs='+', action=ToPathAction, required=True, help='List of paths to your coverage files (bam formatted)')
    io_group.add_argument('--output', type=str, default='output', action=ToPathAction, help='Path to output directory')

    preproc_group = parser.add_argument_group(title='Preprocessing')
    preproc_group.add_argument('--min-ctg-len', type=int, default=2048, help='Minimum contig length')
    preproc_group.add_argument('--min-prevalence', type=int, default=2, help='Minimum contig prevalence for binning. Contig with less that value are filtered out.')
    preproc_group.add_argument('--min-mapping-quality', type=int, default=50, help='Minimum mapping quality for bam filtering')
    preproc_group.add_argument('--flag', type=int, default=3852, help='am Flag for bam filtering')
    preproc_group.add_argument('--fl-range', type=int, default=[200, 500], nargs=2, help='Only allow for paired alignments with spacing within this range')
    preproc_group.add_argument('--tmp-dir', type=str, default='./tmp42', help='Temporary directory for bam processing', action=ToPathAction)

    frag_group = parser.add_argument_group(title='Fragmentation')
    frag_group.add_argument('--fragment-step', type=int, default=128, help='Fragments spacing')
    frag_group.add_argument('--test-ratio', type=float, default=0.1, help='Ratio for train / test split')
    frag_group.add_argument('--n-train', type=int, default=int(1e6), help='Number of training examples')
    frag_group.add_argument('--n-test', type=int, default=int(1e4), help='Number of test examples')

    dl_group = parser.add_argument_group(title='Neural network')
    dl_group.add_argument('--batch-size', type=int, default=256, help='Batch size for training')
    dl_group.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate for gradient descent')
    dl_group.add_argument('--load-batch', type=int, default=200, help='Number of coverage batch to load in memory. Consider lowering this value if your RAM is limited.')
    dl_group.add_argument('--compo-neurons', type=int, default=[64, 32], nargs=2, help='Number of neurons for the composition dense layers (x2)')
    dl_group.add_argument('--cover-neurons', type=int, default=[64, 32], nargs=2, help='Number of neurons for the coverage dense layers (x2)')
    dl_group.add_argument('--cover-filters', type=int, default=32, help='Number of filters for convolution layer of coverage network.')
    dl_group.add_argument('--cover-kernel', type=int, default=7, help='Kernel size for convolution layer of coverage network.')
    dl_group.add_argument('--cover-stride', type=int, default=3, help='Convolution stride for convolution layer of coverage network.')
    dl_group.add_argument('--merge-neurons', type=int, default=32, help='Number of neurons for the merging layer (x1)')
    dl_group.add_argument('--norm', action='store_true', default=False, help='Normalize the k-mer occurrences to frequencies')
    dl_group.add_argument('-k', '--kmer', type=int, default=4, help='k-mer size for composition vector')
    dl_group.add_argument('--no-rc', action='store_true', default=False, help='Do not add the reverse complement k-mer occurrences to the composition vector.')
    dl_group.add_argument('--wsize', type=int, default=64, help='Smoothing window size for coverage vector')
    dl_group.add_argument('--wstep', type=int, default=32, help='Subsampling step for coverage vector')
    dl_group.add_argument('--n-frags', type=int, default=30, help='Number of fragments to split the contigs for the clustering phase')

    cluster_group = parser.add_argument_group(title='Clustering')
    cluster_group.add_argument('--max-neighbors', type=int, default=100, help='Maximum number of neighbors to consider to compute the adjacency matrix.')
    cluster_group.add_argument('--hits-threshold', type=float, default=0.8, help='Minimum percent of edges between two contigs to form an edge between them')
    cluster_group.add_argument('--gamma1', type=float, default=0.1, help='CPM optimization value for the first run of the Leiden clustering')
    cluster_group.add_argument('--gamma2', type=float, default=0.75, help='CPM optimization value for the second run of the Leiden clustering')

    args, unknown = parser.parse_known_args()

    if unknown:
        for key in unknown:
            if key[0] == '-':
                warnings.warn("{} is not a valid argument. It will be ignored".format(key),
                              UserWarning)

    return args
