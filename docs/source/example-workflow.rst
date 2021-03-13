Analysis workflow
-----------------

Here we present a full pipeline example for running CoCoNet from the raw reads to final bins. We assume that you have paired-end illumina reads available in the :code:`data/` directory. Each sample is named with the convention :code:`<sample>_R{1,2}.fastq.gz`.

Trimming
^^^^^^^^

The first step consists in quality trimming and filtering your reads. Many tools can be used for this step, such as `fastp <https://github.com/OpenGene/fastp>`_ or `trimmomatic <http://www.usadellab.org/cms/index.php?page=trimmomatic>`_. Below is a trimming example with fastp. Note that the parameters need to be tuned to your specific case. `FastQC <https://www.bioinformatics.babraham.ac.uk/projects/fastqc>`_ and `MultiQC <https://multiqc.info>`_ can be very useful in that regard.

.. code-block:: bash

    #!/usr/bin/env bash

    # SAMPLES is an environment variable containing the sample names
    # (e.g.: export SAMPLES="sample1 sample2")

    for sample in $SAMPLES; do
        fastp \
        -i data/${sample}_R1.fastq.gz -I data/${sample}_R2.fastq.gz \
        -o ${sample}-trimmed_R1.fastq.gz -O ${sample}-trimmed_R2.fastq.gz
    done


Assembly
^^^^^^^^

To assemble your reads, you have many choices. One of the most accurate for metagenomics is `metaSPAdes <https://cab.spbu.ru/software/meta-spades>`_. However if you have a lot of samples and/or high coverage, metaSPAdes will require a significant amount of time and memory, in which case `Megahit <https://github.com/voutcn/megahit>`_ can be a good alternative.

.. code-block:: bash

   # combine all samples
   cat data/*-trimmed_R1.fastq.gz > forward.fastq.gz
   cat data/*-trimmed_R2.fastq.gz > reverse.fastq.gz

   # run metaspades
   metaspades.py \
   -1 forward.fastq.gz -2 reverse.fastq.gz \
   -o assembly-outputs


Coverage
^^^^^^^^

To compute contig coverage, we can align the trimmed reads on the assembled contigs with tools such as `bwa <http://bio-bwa.sourceforge.net/bwa.shtml>`_ or `bowtie2 <http://bowtie-bio.sourceforge.net/bowtie2/manual.shtml>`_:

.. code-block:: bash

    mkdir coverage

    # Build index
    bwa index -p coverage/index assembly-outputs/scaffolds.fasta

    for sample in $SAMPLES; do
        bwa mem -p coverage/index data/${sample}-trimmed_R*.fastq.gz \
        | samtools view -bh \
        | samtools sort -o coverage/${sample}.bam
        samtools index coverage/${sample}.bam coverage/${sample}.bai
    done


Binning
^^^^^^^

You can finally use CoCoNet and bin your contigs:

.. code-block:: bash

    coconet run --fasta assembly-outputs/scaffolds.fasta --bam coverage/*.bam --output binning

The binning assignment are then available in the file `binning/bins-*.csv`.
