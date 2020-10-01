Analysis workflow
-----------------

Here is an example for running CoCoNet from the raw reads to final bins. Here, we assume that you have paired-end illumina reads are available in the :code:`data/` folder and each sample are named with the convention :code:`<sample>_R1.fastq.gz` if its the forward read and :code:`<sample>_R2.fastq.gz` for the reverse.


Trimming
^^^^^^^^

The first step consists in quality trimming and filtering your reads. Many tools can be used for this step, such as `fastp https://github.com/OpenGene/fastp`_ or `trimmomatic http://www.usadellab.org/cms/index.php?page=trimmomatic`_. Below is a trimming example with fastp. Note that the parameters need to be tuned to your specific case. `FastQC https://www.bioinformatics.babraham.ac.uk/projects/fastqc/`_ and `MultiQC https://multiqc.info/`_ can be very useful in that regard.

.. code-block:: bash

    # SAMPLES is an environment variable containing the sample names (e.g.: export SAMPLES="sample1 sample2")

    for sample in $SAMPLES; do
        fastp \
        -i data/${sample}_R1.fastq.gz -I data/${sample}_R2.fastq.gz \
        -o ${sample}-trimmed_R1.fastq.gz -O ${sample}-trimmed_R2.fastq.gz
    done


Assembly
^^^^^^^^

To assemble your reads, you have a choice many choices. One of the most efficient for metagenomics is `MetaSPAdes https://cab.spbu.ru/software/meta-spades/`_. However if you have a significant amount of samples or coverage, metaSPAdes requires a large amount of time and memory, in which case `Megahit https://github.com/voutcn/megahit`_ can be a good alternative.

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

To compute the contig coverage, we can align the trimmed reads on the assembled contigs with tools such as `bwa http://bio-bwa.sourceforge.net/bwa.shtml`_ or `bowtie2 http://bowtie-bio.sourceforge.net/bowtie2/manual.shtml`_:

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
