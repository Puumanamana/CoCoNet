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

- The binning outcome, `bins_*.csv` with 2 columns: the first is the contig name, and the second is the bin number.
- The run configuration `config.yaml` with the run parameters.
- The filtered alignments (both .bam and .h5 formats) and the filtered assembly (.fasta).
- The list of contigs that were set aside because they didn't pass the prevalence filter, `singletons.txt`. The first column is the contig name, the second is its length, and the remaining ones are the total coverage for each sample.
- Some other files needed to re-run the algorithm from an intermediate step.

Options
^^^^^^^

The latest documentation for all the available running options can be retrieved with the help command:

.. code-block:: bash

    coconet -h

.. program-output:: (echo 'import conf'; tail -n+2 ../../coconet/parser.py; echo 'args=arguments()') | python - --help
   :shell:

usage: coconet [-h] [-n NAME] [-fl FRAGMENT_LENGTH] [-t THREADS]
               [--fasta FASTA] [--coverage COVERAGE [COVERAGE ...]]
               [--output OUTPUT] [--min-ctg-len MIN_CTG_LEN]
               [--min-prevalence MIN_PREVALENCE]
               [--min-mapping-quality MIN_MAPPING_QUALITY] [--flag FLAG]
               [--fl-range FL_RANGE FL_RANGE] [--tmp-dir TMP_DIR]
               [--fragment-step FRAGMENT_STEP] [--test-ratio TEST_RATIO]
               [--n-train N_TRAIN] [--n-test N_TEST] [--batch-size BATCH_SIZE]
               [--learning-rate LEARNING_RATE] [--load-batch LOAD_BATCH]
               [--compo-neurons COMPO_NEURONS COMPO_NEURONS]
               [--cover-neurons COVER_NEURONS COVER_NEURONS]
               [--cover-filters COVER_FILTERS] [--cover-kernel COVER_KERNEL]
               [--cover-stride COVER_STRIDE] [--merge-neurons MERGE_NEURONS]
               [--norm] [-k KMER] [--no-rc] [--wsize WSIZE] [--wstep WSTEP]
               [--n-frags N_FRAGS] [--max-neighbors MAX_NEIGHBORS]
               [--hits-threshold HITS_THRESHOLD] [--gamma1 GAMMA1]
               [--gamma2 GAMMA2]

optional arguments:
  -h, --help            show this help message and exit
  -n NAME, --name NAME  Dataset name
  -fl FRAGMENT_LENGTH, --fragment-length FRAGMENT_LENGTH
                        Dataset name
  -t THREADS, --threads THREADS
                        Number of threads

io:
  --fasta FASTA         Path to your assembly file (fasta formatted)
  --coverage COVERAGE [COVERAGE ...]
                        List of paths to your coverage files (bam formatted)
  --output OUTPUT       Path to output directory

Preprocessing:
  --min-ctg-len MIN_CTG_LEN
                        Minimum contig length
  --min-prevalence MIN_PREVALENCE
                        Minimum contig prevalence for binning. Contig with
                        less that value are filtered out.
  --min-mapping-quality MIN_MAPPING_QUALITY
                        Minimum mapping quality for bam filtering
  --flag FLAG           am Flag for bam filtering
  --fl-range FL_RANGE FL_RANGE
                        Only allow for paired alignments with spacing within
                        this range
  --tmp-dir TMP_DIR     Temporary directory for bam processing

Fragmentation:
  --fragment-step FRAGMENT_STEP
                        Fragments spacing
  --test-ratio TEST_RATIO
                        Ratio for train / test split
  --n-train N_TRAIN     Number of training examples
  --n-test N_TEST       Number of test examples

Neural network:
  --batch-size BATCH_SIZE
                        Batch size for training
  --learning-rate LEARNING_RATE
                        Learning rate for gradient descent
  --load-batch LOAD_BATCH
                        Number of coverage batch to load in memory. Consider
                        lowering this value if your RAM is limited.
  --compo-neurons COMPO_NEURONS COMPO_NEURONS
                        Number of neurons for the composition dense layers
                        (x2)
  --cover-neurons COVER_NEURONS COVER_NEURONS
                        Number of neurons for the coverage dense layers (x2)
  --cover-filters COVER_FILTERS
                        Number of filters for convolution layer of coverage
                        network.
  --cover-kernel COVER_KERNEL
                        Kernel size for convolution layer of coverage network.
  --cover-stride COVER_STRIDE
                        Convolution stride for convolution layer of coverage
                        network.
  --merge-neurons MERGE_NEURONS
                        Number of neurons for the merging layer (x1)
  --norm                Normalize the k-mer occurrences to frequencies
    -k KMER, --kmer KMER  k-mer size for composition vector
  --no-rc               Do not add the reverse complement k-mer occurrences to
                        the composition vector.
  --wsize WSIZE         Smoothing window size for coverage vector
  --wstep WSTEP         Subsampling step for coverage vector
  --n-frags N_FRAGS     Number of fragments to split the contigs for the
                        clustering phase

Clustering:
  --max-neighbors MAX_NEIGHBORS
                        Maximum number of neighbors to consider to compute the
                        adjacency matrix.
  --hits-threshold HITS_THRESHOLD
                        Minimum percent of edges between two contigs to form
                        an edge between them
  --gamma1 GAMMA1       CPM optimization value for the first run of the Leiden
                        clustering
  --gamma2 GAMMA2       CPM optimization value for the second run of the
                        Leiden clustering
