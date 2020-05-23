Usage
-----

Inputs
^^^^^^

CoCoNet's main running mode bins contigs assembled using multiple samples. The minimum required is:

#. The assembly in `fasta` format, generated from the raw reads with your favorite assembler (e.g. `metaspades <https://github.com/ablab/spades>`_, `megahit <https://github.com/voutcn/megahit>`_, `idb-ud <https://github.com/loneknightpy/idba>`_, `metavelvet <https://github.com/hacchy/MetaVelvet>`_, ...)
#. The alignments in the `bam` format. These files can be generated from the raw `fastq` reads and the assembly using an aligner such as `bwa <https://github.com/lh3/bwa>`_ or `bowtie2 <https://github.com/BenLangmead/bowtie2>`_. 

Running CoCoNet
^^^^^^^^^^^^^^^

All of CoCoNet's steps are included in the software, and include:

- *Assembly preprocessing*: Contigs are filtered based on the minimum contig length
- *Coverage preprocessing*: Contigs go through another round of filtering based on the minimum prevalence.
- *Train/Test data generation* for the neural network
- *Deep learning phase*: The neural network is trained de novo for each new dataset to learn the coverage patterns and the k-mer distribution associated present in the data.
- *Clustering phase*: Contigs are placed in bins using the similarity function learned on the previous phase

To run CoCoNet with the default parameters, you simply need to provide the assembly and the bam coverage files:

.. code-block:: bash

   coconet --fasta scaffolds.fasta --coverage cov/*.bam --output binning_results

   
Outputs
^^^^^^^

The output data folder contains many files. The ones you might be interested in are:

- The binning outcome, `bins_*.csv` with 2 columns: the first is the contig name, and the second is the bin number.
- The run configuration `config.yaml` with the run parameters.
- The filtered alignments (both .bam and .h5 formats) and the filtered assembly (.fasta).
- The list of contigs that were set aside because they didn't pass the prevalence filter, `singletons.txt`. The first column is the contig name, the second is its length, and the remaining ones are the total coverage for each sample.
- Some other files needed to re-run the algorithm from an intermediate step.

Options
^^^^^^^

If you want to fine-tune CoCoNet's parameters, you change the parameters using the corresponding flags. The latest documentation for all the available running options can be retrieved with the help command:

.. code-block:: bash

    coconet -h
