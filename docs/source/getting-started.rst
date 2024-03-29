Installation
------------

Pre-requisites
^^^^^^^^^^^^^^

CoCoNet was tested on both MacOS and Ubuntu 18.04.
To install and run CoCoNet, you will need:

#. `python` (>=3.5, recommended: 3.7)
#. `pip3`, the python package manager or the `conda` installer.

If you encounter any issue during the installation, you should retry in a fresh environment (with conda or virtualenv) with the recommended python version (3.7).

Install the latest release on PyPi
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can install CoCoNet, from the `Python Package Index <https://pypi.org/project/coconet-binning/>`_. To install it, you simply need to run the following command (you can omit --user if you're working in a virtual environment).

.. code-block:: bash

    # Install numpy until scikit-bio issue #1690 is resolved
    pip install --user numpy
    # Install CoCoNet
    pip3 install --user coconet-binning


Install the development version
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can also install the most up to date version with the following command:

.. code-block:: bash
                
    # Install numpy until scikit-bio issue #1690 is resolved
    pip install --user numpy
    # Install CoCoNet
    git clone https://github.com/Puumanamana/CoCoNet.git && cd CoCoNet
    pip install --user .

If you encounter any issue with the development version, you should try with the latest release as the development version might not have been thoroughly checked.

Install with bioconda
^^^^^^^^^^^^^^^^^^^^^

CoCoNet is available in bioconda. You will need `miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ or `anaconda <https://anaconda.org/>`_ installed on your computer. CoCoNet can be installed using the following command:

.. code-block:: bash

    # Install in a new environment
    conda create -n coconet -c bioconda -c conda-forge coconet-binning
    # Switch environment
    conda activate coconet

Using Docker
^^^^^^^^^^^^

Alternatively, CoCoNet can be pulled directly from DockerHub. Assuming your contigs are located in /data/contigs.fasta and your indexed bam files are in /data/*.bam and /data/*.bai, then you can run CoCoNet with the following command:

.. code-block:: bash
               
    docker run -v /data:/workspace nakor/coconet coconet run --fasta contigs.fasta --bam *.bam

