.. CoCoNet documentation master file, created by
   sphinx-quickstart on Sat Dec  7 16:20:40 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to CoCoNet's documentation!
===================================

.. image:: https://travis-ci.org/Puumanamana/CoCoNet.svg?branch=master
    :target: https://travis-ci.org/Puumanamana/CoCoNet
.. image:: https://codecov.io/gh/Puumanamana/CoCoNet/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/Puumanamana/CoCoNet
			 
Citation (Work in progress)
---------------------------
Arisdakessian C., Nigro O., Stewart G., Poisson G., Belcaid M.
CoCoNet: An Efficient Deep Learning Tool for Viral Metagenome Binning

Description
-----------

CoCoNet (Composition and Coverage Network) is a binning method for viral metagenomes. It leverages the flexibility and the effectiveness of deep learning models to learn the probability density function of co-occurrence of contigs in the same genome and therefore provides a rigorous probabilistic framework for binning contigs. The derived probability are then used to compute an adjacency matrix for a subset of strategically selected contigs, and infer homogenous clusters representing contigs of the same genome.

Usage
-----

CoCoNet is available in the command line. For a list of all the options, open a terminal and run:

.. code-block:: bash

    python coconet.py run -h

For more details, please see the `documentation <<https://coconet.readthedocs.io/en/latest/index.html>`_

Contribute
----------

 - Issue Tracker: `github <https://github.com/Puumanamana/CoCoNet/issues>`_
 - Source Code: `github <https://github.com/Puumanamana/CoCoNet>`_


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   basic_usage
   subcommands
   examples

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
