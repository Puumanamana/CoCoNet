Installation
------------

Pre-requisites
^^^^^^^^^^^^^^

CoCoNet was tested on both MacOS and Ubuntu 18.04.
To install and run CoCoNet, you will need:

#. `python` (>=3.5)
#. `pip3`, the python package manager
   

Install with bioconda
^^^^^^^^^^^^^^^^^^^^^

CoCoNet is available in bioconda. You will need `miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ or `anaconda <https://anaconda.org/>`_ installed on your computer. CoCoNet can be installed using the following command:

.. code-block:: bash

    conda install -c bioconda coconet-binning              

Install with pip
^^^^^^^^^^^^^^^^

To install CoCoNet, open you need to run (you can omit --user if you're working in a vitualenv):

.. code-block:: bash

    pip3 install coconet-binning --user


Install the development version with pip
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can also install the most up to date version with the following command:

.. code-block:: bash
    git clone https://github.com/Puumanamana/CoCoNet.git
    cd CoCoNet
    pip install .

If you encounter any issue with the development version, you should try with the latest release as the development version might not have been thoroughly checked.
