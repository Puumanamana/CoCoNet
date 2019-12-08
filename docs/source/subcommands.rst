Subcommands
-----------

CoCoNet is composed of multiple subcommands if you only want to perform some part of the analysis.
To display the available subcommands, enter on your terminal

.. code-block:: bash

    python coconet.py -h

Usage: coconet.py [OPTIONS] COMMAND [ARGS]...

Arisdakessian C., Nigro O., Stewart G., Poisson G., Belcaid M.
CoCoNet: An Efficient Deep Learning Tool for Viral Metagenome Binning

Options:
-h, --help  Show this message and exit.  [default: False]

Commands:

preprocess        Preprocess the contig assembly and coverage.
make-train-test   Make train and test examples for neural network.
learn             Train neural network.
cluster           Cluster contigs.
run               Run complete algorithm.

For each subcommand, you can display the list of available parameters by entering

.. code-block:: bash

    python coconet.py SUBCMD -h
