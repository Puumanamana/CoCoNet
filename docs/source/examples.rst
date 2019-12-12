Workflow example
----------------

Here is an example of how to analyze paired-end reads.

Quality filtering and trimming
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First, we need to quality filter the reads. A very popular tool for that is `trimmomatic <http://www.usadellab.org/cms/?page=trimmomatic>`_.
The following code iteratively trims each of your samples by:

#. Removing any leading/trailing base of quality below 3
#. Scanning the read with a 10-bp wide sliding window, cutting when the average quality per base drops below 20
#. Discarding any trimmed reads shorter than 36 bp.

.. code-block:: bash

    mkdir filtered
    for sample in SAMPLES; do
        java -jar trimmomatic.jar PE ${sample}_R1.fastq ${sample}_R2.fastq \
            filtered/${sample}_P_R1.fastq.gz filtered/${sample}_U_R1.fastq.gz \
            filtered/${sample}_P_R2.fastq.gz filtered/${sample}_U_R2.fastq.gz \
            LEADING:3 TRAILING:3 SLIDINGWINDOW:10:20 MINLEN:36
    done
	    
Assembly
^^^^^^^^

You are now ready to assembly your reads. For instance, using `metaspades <https://github.com/ablab/spades>`_:

.. code-block:: bash

   # concatenate the paired samples
   zcat filtered/*_P_R1.fastq.gz > filtered/all_paired_R1.fastq
   zcat filtered/*_P_R2.fastq.gz > filtered/all_paired_R2.fastq
   
   metaspades.py --threads 20 --memory 400 -k 21,33,55,77 \
       -1 filtered/all_paired_R1.fastq -2 filt_paired/all_paired_R2.fastq \
       -s filt_unpaired/all_unpaired.fastq \
       -o metaspades_output 

Coverage
^^^^^^^^

To get your contig coverage, you need to align the filtered reads in each of your samples on the assembly. `bwa <https://github.com/lh3/bwa>`_ can do that for you using the `bwa mem` algorithm: 

.. code-block:: bash

   export THREADS=20

   # First build the index
   bwa index metaspades_output/scaffolds.fasta -p bwa_db

   # Make coverage for each sample
   for sample in SAMPLES; do
       bwa mem -a -M -t $THREADS bwa_db filtered/${sample}_P_R1.fastq.gz filtered/${sample}_P_R2.fastq.gz \
       | samtools sort -@ $THREADS -o coverage/${sample}.bam
   done

Binning your contigs
^^^^^^^^^^^^^^^^^^^^

You are now all set to bin your contigs with `CoCoNet`:

.. code-block:: bash

   coconet run metaspades_output/scaffolds.fasta coverage/*.bam -o binning_results


