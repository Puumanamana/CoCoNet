Usage
-----

Inputs
^^^^^^

CoCoNet's main running mode bins contigs assembled using multiple samples. The minimum required is:

#. The assembly in `fasta` format, generated from the raw reads with your favorite assembler (e.g. `metaspades <https://github.com/ablab/spades>`_, `megahit <https://github.com/voutcn/megahit>`_, `idb-ud <https://github.com/loneknightpy/idba>`_, `metavelvet <https://github.com/hacchy/MetaVelvet>`_, ...)
#. The alignments in the `bam` format. These files can be generated from the raw `fastq` reads and the assembly using an aligner such as `bwa <https://github.com/lh3/bwa>`_ or `bowtie2 <https://github.com/BenLangmead/bowtie2>`_. 


Running CoCoNet
^^^^^^^^^^^^^^^

The CoCoNet package comes with 4 subcommands to run different part of the algorithm:

- :code:`preprocess`: Filter the assembly and coverage based on minimum length, and prevalence. In addition, alignments are filtered based on quality, SAM flag, coverage and template length.
- :code:`learn`: Train the neural network to predict whether two DNA fragments belong to the same genome.
- :code:`cluster`: Cluster the contigs using the previously trained neural network.
- :code:`run`: Runs all the previous steps combined (recommended)

To run CoCoNet with the default parameters, you simply need to provide the assembly and the bam coverage files:

.. code-block:: bash

   coconet --fasta scaffolds.fasta --bam cov/*.bam --output binning_results


You can see the usage information for each subcommand by typing :code:`coconet <subcommand> -h`, where `<subcommand>` is either `preprocess`, `learn`, `cluster` or `run`. For more details about the options, see the :ref:`hyperparameters <hyperparameters>` section

You can use the :code:`--continue` flag to resume an interrupted run. However, depending where your run was interrupted, you might have corrupted files, in
which case you would need to either re-run from the start or delete the corrupted file.

Outputs
^^^^^^^

The output data folder contains many files. The ones you might be interested in are:

- The binning outcome, `bins_*.csv` with 2 columns: the first is the contig name, and the second is the bin number.
- The log file, `coconet.log`, that contains all of the runtime information, run parameters and filtering information.  
- The run configuration `config.yaml` with the run parameters.
- The filtered alignments (both .bam and .h5 formats) and the filtered assembly (.fasta).
- The list of contigs that were set aside because they didn't pass the prevalence or minimum length filter, `exclude.tsv`. The first column is the contig name, the second is the reason why the contig was excluded, and the remaining field is the values corresponding to the filtering criteria (length or list of coverage values).
