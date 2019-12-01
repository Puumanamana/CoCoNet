!/usr/bin/env nextflow

nextflow.preview.dsl=2

process CoCoNet_runner {
    label "high_computation"
    tag "${folder}"
    publishDir CoCoNet_results

    input:
    file folder

    output:
    file "output_data"

    script:
    """
    python ${script_dir}/main.py --name ${folder}
    """
}

INPUT_FOLDER = Channel.fromPath(params.input_dir)
// COVERAGE_FILES = Channel.fromPath("${params.input_dir}/*/coverage_contigs_gt2048_prev2.h5.gz")
// ASSEMBLY_FILES = Channel.fromPath("${params.input_dir}/*/assembly_gt2048_prev2.fasta")

workflow {
    main:
    INPUT_FOLDER | CoCoNet_runner
}

