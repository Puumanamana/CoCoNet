'''
CoCoNet parser information and documentation for click
'''

from pathlib import Path
import click

orig_init = click.core.Option.__init__

def new_init(self, *args, **kwargs):
    '''
    Show default values in the help documentation
    '''
    orig_init(self, *args, **kwargs)
    self.show_default = True

click.core.Option.__init__ = new_init

def to_path(_ctx, _param, value):
    '''
    Converts path from command line to Path objects
    '''
    if isinstance(value, (list, tuple)):
        return [Path(val) for val in value]
    if isinstance(value, str):
        return Path(value)
    return value

_HELP_INPUTS = '''
Positional arguments:

<fasta>      Path to your assembly file (fasta formatted)

<coverage>   List of paths to your coverage files (bam formatted)
'''

_HELP_MSG = {
    'preprocess': 'Preprocess the contig assembly and coverage.\n{}'.format(_HELP_INPUTS),
    'make_train_test': 'Make train and test examples for neural network.\n{}'.format(_HELP_INPUTS),
    'learn': 'Train neural network.\n{}'.format(_HELP_INPUTS),
    'cluster': 'Cluster contigs.\n{}'.format(_HELP_INPUTS),
    'run': 'Run complete algorithm.\n{}'.format(_HELP_INPUTS)
}

_OPTIONS = {
    'io': [
        click.argument('fasta', required=False, callback=to_path, type=click.Path(exists=True)),
        click.argument('coverage', required=False, nargs=-1, callback=to_path, type=click.Path(exists=True)),
        click.option('-o', '--output', type=str, required=False, default='output', callback=to_path,
                     help='Path to output directory'),
    ],

    'general': [
        click.option('-n', '--name', type=str, required=False, default='ds',
                     help='Dataset name'),
        click.option('-fl', '--fragment-length', type=int, required=False, default=1024,
                     help='Fragment length for contig splitting'),
        click.option('-t', '--threads', type=int, required=False, default=30,
                     help='Number of threads'),
    ],

    'core': [
    ],

    'preproc': [
        click.option('--min-ctg-len', type=int, required=False, default=2048,
                     help='Minimum contig length'),
        click.option('--min-prevalence', type=int, required=False, default=2,
                     help='Minimum contig prevalence for binning. Contig with less that value are filtered out.'),
        click.option('--min-mapping-quality', type=click.IntRange(1, 60), required=False, default=50,
                     help='Minimum mapping quality for bam filtering'),
        click.option('--flag', type=int, required=False, default=3596,
                     help='Sam Flag for bam filtering'),
        click.option('--fl-range', type=int, required=False, default=(200, 500), nargs=2,
                     help='Only allow for paired alignments with spacing within this range'),
        click.option('--tmp-dir', type=str, required=False, default='./tmp42', callback=to_path,
                     help='Temporary directory for bam processing'),
    ],

    'frag': [
        click.option('--fragment-step', type=int, required=False, default=128,
                     help='Fragments spacing'),
        click.option('--test-ratio', type=click.FloatRange(.1, .99), required=False, default=0.1,
                     help='Ratio for train / test split'),
        click.option('--n-train', type=int, required=False, default=int(1e6),
                     help='Number of training examples'),
        click.option('--n-test', type=int, required=False, default=int(1e4),
                     help='Number of test examples')
    ],

    'dl': [
        click.option('--batch-size', type=int, required=False, default=256,
                     help='Batch size for training'),
        click.option('--learning-rate', type=float, required=False, default=1e-4,
                     help='Learning rate for gradient descent'),
        click.option('--load-batch', type=int, required=False, default=200,
                     help='Number of coverage batch to load in memory. Consider lowering this value if your RAM is limited.'),
        click.option('--compo-neurons', type=int, required=False, default=(64, 32), nargs=2,
                     help='Number of neurons for the composition network (2 layers)'),
        click.option('--cover-neurons', type=int, required=False, default=(64, 32), nargs=2,
                     help='Number of neurons for the coverage network (2 layers)'),
        click.option('--cover-filters', type=int, required=False, default=32,
                     help='Number of filters for convolution layer of coverage network.'),
        click.option('--cover-kernel', type=int, required=False, default=7,
                     help='Kernel size for convolution layer of coverage network.'),
        click.option('--cover-stride', type=int, required=False, default=3,
                     help='Convolution stride for convolution layer of coverage network.'),
        click.option('--merge-neurons', type=int, required=False, default=32,
                     help='Number of neurons for the merging network (1 layer)'),
        click.option('--norm', required=False, is_flag=True,
                     help='Normalize the k-mer occurrences to frequencies'),
        click.option('-k', '--kmer', type=int, required=False, default=4,
                     help='k-mer size for composition vector'),
        click.option('--no-rc', required=False, is_flag=True,
                     help='Do not add the reverse complement k-mer occurrences to the composition vector'),
        click.option('--wsize', type=int, required=False, default=64,
                     help='Smoothing window size for coverage vector'),
        click.option('--wstep', type=int, required=False, default=32,
                     help='Subsampling step for coverage vector'),
        click.option('--n-frags', type=int, required=False, default=30,
                     help='Number of fragments to split the contigs for the clustering phase')
    ],

    'cluster': [
        click.option('--max-neighbors', type=int, required=False, default=100,
                     help='Maximum number of neighbors to consider to compute the adjacency matrix.'),
        click.option('--hits-threshold', type=click.FloatRange(0, 1), required=False, default=0.8,
                     help='Minimum percent of edges between two contigs to form an edge between them.'),
        click.option('--gamma1', type=click.FloatRange(0, 1), required=False, default=0.1,
                     help='CPM optimization value for the first run of the Leiden clustering'),
        click.option('--gamma2', type=click.FloatRange(0, 1), required=False, default=0.75,
                     help='CPM optimization value for the second run of the Leiden clustering'),
    ]
}

def make_decorator(*keys):
    '''
    Combines all options in list in a single decorator
    '''

    def add_opt(func):
        for key in keys:
            for option in reversed(_OPTIONS[key]):
                func = option(func)
        return func
    return add_opt
