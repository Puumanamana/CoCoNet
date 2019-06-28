#!/usr/bin/env nextflow

inputdir = 'camisim/sim/*_sample_*/bam/*.bam'

alignments = Channel.fromPath(inputdir)
    .ifEmpty { error "Cannot find any bam matching: ${inputdir}" }
    .map{ path -> tuple( (path =~ "sample_[0-9]+")[0], path )  }

genome_lengths = Channel
    .fromPath('/home/cedric/database/viral.genomic.ACGT.fasta')
    .splitFasta( record: [id: true, seqString: true ])
    .map{ it -> tuple( it.id, it.seqString.length() ) }

process Samtools {

    input:
	set val(sample), file(bam) from alignments
    output:
        set stdout, file("*.txt") into depth_txt

    script:
    """
    #!/usr/bin/env bash

    ctg_name=`basename ${bam} .bam`

    samtools depth ${bam} > "${sample}_\$ctg_name.txt"

    printf \$ctg_name
    """

}

process TxtToNpy {
    
    input:
        set val(contig), val(genome_len), file(f) from genome_lengths.join(depth_txt.groupTuple())
    output:
	file("*.npy") into depth_npy

    script:
    """
    #!/usr/bin/env python3

    import pandas as pd
    import numpy as np

    files = [ "${f.join('","')}" ]

    datasets = [ pd.read_csv(f, header=None, sep='\\t', names=["acc","pos","depth"]) 
                 for f in files ]

    coverage = np.zeros([${genome_len}, ${f.size()}],dtype=np.uint32)

    for i,dataset in enumerate(datasets):
        if dataset.size > 0:
            coverage[dataset.pos.values-1,i] = dataset.depth.values

    np.save("${contig}.npy",coverage)
    """
}

process NpyToH5 {

    publishDir "${PWD}/coverage_nxf", mode: "copy"
    
    input:
        file f from depth_npy.collect()
    output:
	file("coverage.h5")

    script:
    """
    #!/usr/bin/env python3

    from glob import glob
    import numpy as np
    import h5py

    cov_h5 = h5py.File("coverage.h5","w")
    
    for filename in glob("*.npy"):
        virus = filename.split(".")[0]
        matrix = np.load(filename)
        cov_h5.create_dataset(virus,data=matrix)
    cov_h5.close()
    """
}

