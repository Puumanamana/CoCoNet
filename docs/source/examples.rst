Running coconet
---------------

To run coconet, you will need to:

- Quality filter your reads
- Use your favorite metagenomics assembler to build contigs
- Compute the coverage by alignning your reads on your assembled contigs

Binning your contigs
^^^^^^^^^^^^^^^^^^^^

You are now all set to bin your contigs with `CoCoNet`:

.. code-block:: bash

   coconet --fasta metaspades_output/scaffolds.fasta --coverage coverage/*.bam --output binning_results


