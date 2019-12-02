N_GENOMES = 2000 500
N_COVERAGES = 3 10
N_SAMPLES = 4 15
N_ITER = 0 1 2 3 4 5 6 7 8 9

test:
	python tests/test_generators.py
	python tests/test_fragmentation.py

paper:
	@$(foreach g, $(N_GENOMES), \
	$(foreach c, $(N_COVERAGES), \
	$(foreach s, $(N_SAMPLES), \
	$(foreach i, $(N_ITER), \
	    $(eval rundir="${g}_${c}_${s}_${i}") \
	    echo ${rundir} && \
	    python main.py run input_data/${rundir}/assembly.fasta input_data/${rundir}/coverage_contigs_gt2048.h5 --output=output_data/${rundir} && )))) :

