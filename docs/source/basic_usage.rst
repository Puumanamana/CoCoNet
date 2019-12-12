Basic Usage
-----------

Inputs
^^^^^^

CoCoNet's main running mode bins contigs assembled using multiple samples. The minimum required is:

#. The assembly in `fasta` format, generated from the raw reads with your favorite assembler (e.g. `metaspades <https://github.com/ablab/spades>`_, `megahit <https://github.com/voutcn/megahit>`_, `idb-ud <https://github.com/loneknightpy/idba>`_, `metavelvet <https://github.com/hacchy/MetaVelvet>`_, ...)
#. The alignments in the `bam` format. These files can be generated from the raw `fastq` reads and the assembly using an aligner such as `bwa <https://github.com/lh3/bwa>`_ or `bowtie2 <https://github.   com/BenLangmead/bowtie2>`_. 

Outputs
^^^^^^^

The output data folder contains many files. The ones you might be interested in are:

- The binning outcome, `final_bins_*.csv` with 2 columns: the first is the contig name, and the second is the bin number.
- The run configuration `config.yaml` with the run parameters.
- The filtered alignments (both .bam and .h5 formats) and the filtered assembly (.fasta).
- The list of contigs that were set aside because they didn't pass the prevalence filter, `singletons.txt`. The first column is the contig name, the second is its length, and the remaining ones are the total coverage for each sample.
- Some other files needed to re-run the algorithm from an intermediate step.

Options
^^^^^^^
   
The latest documentation for all the available running options can be retrieved with the help command:

.. code-block:: bash

    python coconet.py run -h
   
Usage: coconet run [OPTIONS] [FASTA] [COVERAGE]...

  Run complete algorithm.

Positional arguments:

============   ====================================================
Name           Description
============   ====================================================
`<fasta>`      Path to your assembly file (fasta formatted)
`<coverage>`   List of paths to your coverage files (bam formatted)
============   ====================================================

Options:

-o, --output TEXT               Path to output directory  [default: output]
-n, --name TEXT                 Dataset name  [default: ds]
-fl, --fragment-length INTEGER  Fragment length for contig splitting [default: 1024]
-t, --threads INTEGER           Number of threads  [default: 30]
--min-ctg-len INTEGER           Minimum contig length  [default: 2048]
--min-prevalence INTEGER        Minimum contig prevalence for binning. Contig with less that value are filtered out.  [default: 2]
--min-mapping-quality INTEGER   Minimum mapping quality for bam filtering [default: 50]
--flag INTEGER                  Sam Flag for bam filtering  [default: 3596]
--fl-range INTEGER              Only allow for paired alignments with spacing within this range  [default: 200, 500]
--tmp-dir TEXT                  Temporary directory for bam processing [default: ./tmp42]
--fragment-step INTEGER         Fragments spacing  [default: 128]
--test-ratio FLOAT              Ratio for train / test split  [default: 0.1]
--n-train INTEGER               Number of training examples  [default: 1000000]
--n-test INTEGER                Number of test examples  [default: 10000]
--batch-size INTEGER            Batch size for training  [default: 256]
--learning-rate FLOAT           Learning rate for gradient descent [default: 0.0001]
--load-batch INTEGER            Number of coverage batch to load in memory. Consider lowering this value if your RAM is Consider lowering this value if your RAM is
--compo-neurons INTEGER         Number of neurons for the composition network (2 layers)  [default: 64, 32]
--cover-neurons INTEGER         Number of neurons for the coverage network  (2 layers)  [default: 64, 32]
--cover-filters INTEGER         Number of filters for convolution layer of coverage network.  [default: 32]
--cover-kernel INTEGER          Kernel size for convolution layer of coverage network.  [default: 7]
--cover-stride INTEGER          Convolution stride for convolution layer of coverage network.  [default: 7]
--merge-neurons INTEGER         Number of neurons for the merging network (1 layer)  [default: 32]
--norm                          Normalize the k-mer occurrences to frequencies  [default: False]
-k, --kmer INTEGER              k-mer size for composition vector  [default: 4]
--no-rc                         Do not add the reverse complement k-mer occurrences to the composition vector occurrences to the composition vector
--wsize INTEGER                 Smoothing window size for coverage vector [default: 64]
--wstep INTEGER                 Subsampling step for coverage vector [default: 32]
--n-frags INTEGER               Number of fragments to split the contigs for the clustering phase  [default: 30]
--max-neighbors INTEGER         Maximum number of neighbors to consider to compute the adjacency matrix.  [default: 100]
--hits-threshold FLOAT          Minimum percent of edges between two contigs to form an edge between them.  [default: 0.8]
--gamma1 FLOAT                  CPM optimization value for the first run of the Leiden clustering  [default: 0.1]
--gamma2 FLOAT                  CPM optimization value for the second run of the Leiden clustering  [default: 0.75]
-h, --help                      Show this message and exit.  [default: False]
