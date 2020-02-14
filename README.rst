CoCoNet documentation
=====================

.. image:: https://travis-ci.org/Puumanamana/CoCoNet.svg?branch=master
    :target: https://travis-ci.org/Puumanamana/CoCoNet
.. image:: https://codecov.io/gh/Puumanamana/CoCoNet/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/Puumanamana/CoCoNet
.. image:: https://readthedocs.org/projects/coconet/badge/?version=latest
    :target: https://coconet.readthedocs.io/en/latest/?badge=latest
.. image:: https://api.codacy.com/project/badge/Grade/552eeafebb52496ebb409f421bd4edb6
    :target: https://www.codacy.com/manual/Puumanamana/CoCoNet?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=Puumanamana/CoCoNet&amp;utm_campaign=Badge_Grade			 

Citation (Work in progress)
---------------------------
Arisdakessian C., Nigro O., Steward G., Poisson G., Belcaid M.
CoCoNet: An Efficient Deep Learning Tool for Viral Metagenome Binning

Description
-----------

CoCoNet (Composition and Coverage Network) is a binning method for viral metagenomes. It leverages the flexibility and the effectiveness of deep learning models to learn the probability density function of co-occurrence of contigs in the same genome and therefore provides a rigorous probabilistic framework for binning contigs. The derived probability are then used to compute an adjacency matrix for a subset of strategically selected contigs, and infer homogenous clusters representing contigs of the same genome.

Install
-------

CoCoNet is available on PyPi and can be installed with pip:

.. code-block:: bash

   pip3 install coconet-binning --user

Usage
-----

CoCoNet is available in the command line. For a list of all the options, open a terminal and run:

.. code-block:: bash

    coconet -h

For more details, please see the documentation on `ReadTheDocs <https://coconet.readthedocs.io/en/latest/index.html>`_

Contribute
----------

- Issue Tracker: `github <https://github.com/Puumanamana/CoCoNet/issues>`__
- Source Code: `github <https://github.com/Puumanamana/CoCoNet>`__
