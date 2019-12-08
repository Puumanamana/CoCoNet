CoCoNet 1.0.0
=============
.. image:: https://travis-ci.org/Puumanamana/CoCoNet.svg?branch=master
    :target: https://travis-ci.org/Puumanamana/CoCoNet
.. image:: https://codecov.io/gh/Puumanamana/CoCoNet/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/Puumanamana/CoCoNet

Citation (Work in progress)
--------
Arisdakessian C., Nigro O., Stewart G., Poisson G., Belcaid M.
CoCoNet: An Efficient Deep Learning Tool for Viral Metagenome Binning

Documentation
-------------
Detailed documentation is available on `ReadTheDocs <https://coconet.readthedocs.io/en/latest/index.html>`_

Installation
------------
To install CoCoNet, open a terminal and run (you can omit --user if you're working in a vitualenv):
.. code-block:: bash

    pip3 install coconet-binning --user

Usage
-----
You can run CoCoNet with the command line. There are two required arguments:
- The assembly in the *fasta* format
- The coverage files in the *bam* format

Running the complete algorithm
------------------------------

You can process your dataset with CoCoNet using the following command:
.. code-block:: bash

    python coconet.py [ASSEMBLY] [COV_SAMPLE_1] [COV_SAMPLE_2] ... [COV_SAMPLE_N]

To see all the available options, enter :bash:``python coconet.py run -h``Detailed documentation is available on `ReadTheDocs <https://coconet.readthedocs.io/en/latest/index.html>`_:

Usage: coconet.py run [OPTIONS] [FASTA] [COVERAGE]...

Run complete algorithm.

[FASTA] Path to your assembly file (fasta formatted)
[COVERAGE] List of paths to your coverage files (bam formatted)

Options:
  --n-frags INTEGER               Number of fragments to split a contigs
                                  [default: 30]
  --max-neighbors INTEGER         Maximum number of neighbors to consider to
                                  compute the adjacency matrix.  [default:
                                  100]
  --hits-threshold FLOAT RANGE    Minimum percent of edges between two contigs
                                  to form an edge between them.  [default:
                                  0.8]
  --gamma1 FLOAT RANGE            CPM optimization value for the first run of
                                  the Leiden clustering  [default: 0.1]
  --gamma2 FLOAT RANGE            CPM optimization value for the second run of
                                  the Leiden clustering  [default: 0.75]
  --batch-size INTEGER            Batch size for training  [default: 256]
  --learning-rate FLOAT           Learning rate for gradient descent
                                  [default: 0.0001]
  --load-batch INTEGER            Number of coverage batch to load in memory.
                                  Consider lowering this value if your RAM is
                                  limited.  [default: 500]
  --compo-neurons INTEGER...      Number of neurons for the composition
                                  network (2 layers)  [default: 64, 32]
  --cover-neurons INTEGER...      Number of neurons for the coverage network
                                  (2 layers)  [default: 64, 32]
  --cover-filters INTEGER         Number of filters for convolution layer of
                                  coverage network.  [default: 32]
  --cover-kernel INTEGER          Kernel size for convolution layer of
                                  coverage network.  [default: 7]
  --cover-stride INTEGER          Convolution stride for convolution layer of
                                  coverage network.  [default: 3]
  --combined-neurons INTEGER      Number of neurons for the merging network (1
                                  layer)  [default: 32]
  --norm                          Normalize the k-mer occurrences to
                                  frequencies  [default: False]
  -k, --kmer INTEGER              k-mer size for composition vector  [default:
                                  4]
  --no-rc                         Do not add the reverse complement k-mer
                                  occurrences to the composition vector
                                  [default: False]
  --wsize INTEGER                 Smoothing window size for coverage vector
                                  [default: 64]


Running specific steps
----------------------

CoCoNet is composed of multiple subcommands if you only want to perform some part of the analysis.
To display the documentation for each subcommands, enter on your terminal :bash:`python coconet.py -h`

Usage: coconet.py [OPTIONS] COMMAND [ARGS]...

Arisdakessian C., Nigro O., Stewart G., Poisson G., Belcaid M. CoCoNet: An
Efficient Deep Learning Tool for Viral Metagenome Binning

Options:
  -h, --help  Show this message and exit.  [default: False]

Commands:
  preprocess       Preprocess the contig assembly and coverage.
  make-train-test  Make train and test examples for neural network.
  learn            Train neural network.
  cluster          Cluster contigs.
  run              Run complete algorithm.

For each subcommand, you can display the list of available parameters by entering :bash:`python coconet.py SUBCMD -h`.

Contribute
----------

 - Issue Tracker: `github <https://github.com/Puumanamana/CoCoNet/issues>`_
 - Source Code: `github <https://github.com/Puumanamana/CoCoNet>`_
