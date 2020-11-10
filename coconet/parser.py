"""
CoCoNet parser information and documentation
"""

from pathlib import Path
import argparse
import logging
import os


SUB_COMMANDS = ['preprocess', 'learn', 'cluster']

def get_version():
    """
    Return CoCoNet version
    """

    from coconet import __version__
    return 'CoCoNet v{version}'.format(version=__version__)


class ToPathAction(argparse.Action):
    """
    argparse action to convert string to Path objects
    """

    def __init__(self, option_strings, dest, required=False, **kwargs):

        argparse.Action.__init__(self,
                                 option_strings=option_strings,
                                 dest=dest,
                                 **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        if isinstance(values, (list, tuple)):
            values_ok = [Path(val).resolve() for val in values]
        else:
            values_ok = Path(values).resolve()

        setattr(namespace, self.dest, values_ok)

def parse_args():
    """
    Command line parser for CoCoNet algorithm
    """

    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--version', action='version', version=get_version()
    )

    #========================================================#
    #=========== General arguments of any program ===========#
    #========================================================#

    main_parser = argparse.ArgumentParser(add_help=False)
    main_parser.add_argument(
        '--output', type=str, default='output', action=ToPathAction,
        help='Path to output directory'
    )
    main_parser.add_argument(
        '-t', '--threads', type=int, default=5,
        help='Number of threads'
    )
    main_parser.add_argument(
        '--debug', action='store_const', dest='loglvl', const=logging.DEBUG, default=logging.INFO,
        help='Print debugging statements'
    )
    main_parser.add_argument(
        '--quiet', action='store_const', dest='loglvl', const=logging.WARNING,
        help='Less verbose'
    )
    main_parser.add_argument(
        '--silent', action='store_const', dest='loglvl', const=logging.ERROR,
        help='Only error messages'
    )
    main_parser.add_argument(
        '--continue', action='store_true',
        help='Start from last checkpoint. The output directory needs to be the same.'
    )

    #========================================================#
    #================== Path to input data ==================#
    #========================================================#

    input_parser = argparse.ArgumentParser(add_help=False)
    input_parser.add_argument(
        '--fasta', type=str, action=ToPathAction,
        help='Path to your assembly file (fasta formatted)'
    )
    input_parser.add_argument(
      '--h5', type=str, action=ToPathAction,
        help=('Experimental: coverage in hdf5 format '
              '(keys are contigs, values are (sample, contig_len) ndarrays')
    )

    #========================================================#
    #============ Generic parameters for CoCoNet ============#
    #========================================================#

    global_parser = argparse.ArgumentParser(add_help=False)
    global_parser.add_argument(
        '--fragment-length', type=int, default=-1,
        help=('Length of contig fragments in bp. '
              'Default is half the minimum contig length.')
    )
    global_parser.add_argument(
        '--features', type=str, default=['coverage', 'composition'],
        choices=['coverage', 'composition'], nargs='+',
        help='Features for binning (composition, coverage, or both)'
    )

    #========================================================#
    #================= Preprocessing parser =================#
    #========================================================#

    preproc_parser = argparse.ArgumentParser(add_help=False)
    preproc_parser.add_argument(
        '--bam', type=str, nargs='+', action=ToPathAction,
        help='List of paths to your coverage files (bam formatted)'
    )

    preproc_parser.add_argument(
        '--min-ctg-len', type=int, default=2048,
        help='Minimum contig length'
    )
    preproc_parser.add_argument(
        '--min-prevalence', type=int, default=2,
        help=('Minimum contig prevalence for binning. '
              'Contig with less that value are filtered out.')
    )
    preproc_parser.add_argument(
        '--min-mapping-quality', type=int, default=30,
        help='Minimum alignment quality'
    )
    preproc_parser.add_argument(
        '--min-aln-coverage', type=float, default=50,
        help='Discard alignments with less than %(default)s%% aligned nucleotides'
    )
    preproc_parser.add_argument(
        '--flag', type=int, default=3596,
        help='SAM flag for filtering (same as samtools "-F" option)'
    )
    preproc_parser.add_argument(
        '--tlen-range', type=int, nargs=2,
        help='Only allow for paired alignments with spacing within this range'
    )
    preproc_parser.add_argument(
        '--min-dtr-size', type=int, default=10,
        help='Minimum size of DTR to flag complete contigs'
    )

    #========================================================#
    #=============== Subparser: deep learning ===============#
    #========================================================#

    dl_parser = argparse.ArgumentParser(add_help=False)

    dl_parser.add_argument(
        '--fragment-step', type=int, default=128,
        help='Fragments spacing'
    )
    dl_parser.add_argument(
        '--test-ratio', type=float, default=0.1,
        help='Ratio for train / test split'
    )
    dl_parser.add_argument(
        '--n-train', type=int, default=int(4e6),
        help='Maximum number of training examples'
    )
    dl_parser.add_argument(
        '--n-test', type=int, default=int(1e4),
        help='Number of test examples'
    )
    dl_parser.add_argument(
        '--learning-rate', type=float, default=1e-3,
        help='Learning rate for gradient descent'
    )
    dl_parser.add_argument(
        '--batch-size', type=int, default=256,
        help='Batch size for training'
    )
    dl_parser.add_argument(
        '--test-batch', type=int, default=400,
        help='Run test every %(default)s batches'
    )
    dl_parser.add_argument(
        '--patience', type=int, default=5,
        help='Early stopping if test accuracy does not improve for %(default)s consecutive tests'
    )
    dl_parser.add_argument(
        '--load-batch', type=int, default=100,
        help=('Number of coverage batch to load in memory. '
              'Consider lowering this value if your RAM is limited.')
    )
    dl_parser.add_argument(
        '--compo-neurons', type=int, default=[64, 32], nargs=2,
        help='Number of neurons for the composition dense layers (x2)'
    )
    dl_parser.add_argument(
        '--cover-neurons', type=int, default=[64, 32], nargs=2,
        help='Number of neurons for the coverage dense layers (x2)'
    )
    dl_parser.add_argument(
        '--cover-filters', type=int, default=16,
        help='Number of filters for convolution layer of coverage network.'
    )
    dl_parser.add_argument(
        '--cover-kernel', type=int, default=4,
        help='Kernel size for convolution layer of coverage network.'
    )
    dl_parser.add_argument(
        '--cover-stride', type=int, default=2,
        help='Convolution stride for convolution layer of coverage network.'
    )
    dl_parser.add_argument(
        '--merge-neurons', type=int, default=32,
        help='Number of neurons for the merging layer (x1)'
    )
    dl_parser.add_argument(
        '-k', '--kmer', type=int, default=4,
        help='k-mer size for composition vector'
    )
    dl_parser.add_argument(
        '--no-rc', action='store_true', default=False,
        help='Do not add the reverse complement k-mer occurrences to the composition vector.'
    )
    dl_parser.add_argument(
        '--wsize', type=int, default=64,
        help='Smoothing window size for coverage vector'
    )
    dl_parser.add_argument(
        '--wstep', type=int, default=32,
        help='Subsampling step for coverage vector'
    )
    dl_parser.add_argument(
        '--n-frags', type=int, default=30,
        help='Number of fragments to split the contigs for the clustering phase'
    )

    #========================================================#
    #====================== Clustering ======================#
    #========================================================#

    cluster_parser = argparse.ArgumentParser(add_help=False)

    cluster_parser.add_argument(
        '--max-neighbors', type=int, default=250,
        help='Maximum number of neighbors to consider to compute the adjacency matrix.'
    )
    cluster_parser.add_argument(
        '--vote-threshold', type=float, default=None,
        help=('When this parameter is not set, contig-contig edges are computed '
              'by summing the probability between all pairwise fragments between them.'
              'Otherwise, adopt a voting strategy and sets a hard-threshold on the probability'
              'from each pairwise comparison.')
    )
    cluster_parser.add_argument(
        '--algorithm', type=str, default='leiden',
        choices=['leiden', 'spectral'],
        help=('Algorithm for clustering the contig-contig graph. '
              'Note: the number of cluster is required if "spectral" is chosen.')
    )
    cluster_parser.add_argument(
        '--theta', type=float, default=0.8,
        help='(leiden) Minimum percent of edges between two contigs to form an edge between them'
    )
    cluster_parser.add_argument(
        '--gamma1', type=float, default=0.3,
        help='(leiden) CPM optimization value for the first run of the Leiden clustering'
    )
    cluster_parser.add_argument(
        '--gamma2', type=float, default=0.4,
        help='(leiden) CPM optimization value for the second run of the Leiden clustering'
    )
    cluster_parser.add_argument(
        '--n-clusters', type=int,
        help='(spectral clustering) Maximum number of clusters'
    )
    cluster_parser.add_argument(
        '--recruit-small-contigs', action='store_true',
        help='Salvage short contigs (<2048)'
    )
    #========================================================#
    #====================== Subparsers ======================#
    #========================================================#

    subparsers = parser.add_subparsers(title='action', dest='action')
    subparsers.add_parser(
        'preprocess', parents=[input_parser, main_parser, preproc_parser],
        help='Preprocess data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    subparsers.add_parser(
        'learn', parents=[input_parser, main_parser, global_parser, dl_parser],
        help='Train neural network on input data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    subparsers.add_parser(
        'cluster', parents=[main_parser, global_parser, cluster_parser],
        help='Bin contigs using neural network',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    subparsers.add_parser(
      'run',
      parents=[input_parser, main_parser, preproc_parser,
               dl_parser, cluster_parser, global_parser],
      help='Run complete workflow (recommended)',
      formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    args, _ = parser.parse_known_args()

    if args.action is None:
        return parser.parse_known_args(['run'])[0]

    if (hasattr(args, 'algorithm')
        and args.algorithm == 'spectral'
        and not isinstance(args.n_clusters, int)):
        logging.warning('--n-clusters needs to be set when --algorithm is "spectral"')
        raise ValueError

    os.environ['COCONET_CONTINUE'] = 'Y' if getattr(args, 'continue') else 'N'

    return args
