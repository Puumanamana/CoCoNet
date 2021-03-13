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

CoCoNet (Composition and Coverage Network) is a binning method for viral metagenomes. It leverages deep learning to abstract the modeling of the k-mer composition and the coverage for binning contigs assembled form viral metagenomic data. Specifically, our method uses a neural network to learn from the metagenomic data a flexible function for predicting the probability that any pair of contigs originated from the same genome. These probabilities are subsequently combined to infer bins, or clusters representing the species present in the sequenced samples. Our approach was specifically designed for diverse viral metagenomes, such as those found in environmental samples (e.g., oceans, soil, etc.).

Install
-------

Install latest PyPi release (recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pip3 install --user numpy
   pip3 install --user coconet-binning

For more installation options, see the `documentation <https://coconet.readthedocs.io/getting-started.html>`_
   
Basic usage
-----------

CoCoNet is available as the command line program. For a list of all the options, open a terminal and run:

.. code-block:: bash

    coconet run -h

For more details, please see the documentation on `ReadTheDocs <https://coconet.readthedocs.io>`_

Checking the installation
-------------------------

A test dataset is provided in this repository in tests/sim_data. To quickly verify the installation worked, you can simply download the repository and run the test command as follows:

.. code-block:: bash

   git clone https://github.com/Puumanamana/CoCoNet
   cd CoCoNet
   make test

Contribute
----------

- Issue Tracker: `github <https://github.com/Puumanamana/CoCoNet/issues>`__
- Source Code: `github <https://github.com/Puumanamana/CoCoNet>`__
