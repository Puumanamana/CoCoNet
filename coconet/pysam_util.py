import pysam
import h5py
import pandas as pd
import numpy as np


def pass_filter(s, min_mapq=30, min_tlen=None, max_tlen=None, min_coverage=0.5):
    if s.mapping_quality < min_mapq:
        return False
    if s.is_unmapped:
        return False
    if not s.is_paired:
        return False
    if s.is_duplicate:
        return False
    if s.is_qcfail:
        return False
    if s.is_secondary:
        return False
    if s.is_supplementary:
        return False
    if s.query_alignment_length / s.query_length < min_coverage:
        return False
    if min_tlen is not None and abs(s.template_length) < min_tlen:
        return False
    if max_tlen is not None and abs(s.template_length) > max_tlen:
        return False
    return True
    
def get_contig_coverage(iterator, length):
    coverage = np.zeros(length)
    
    for read in filter(pass_filter, iterator):
        # Need to handle overlap between forward and reverse read
        coverage[read.reference_start+1:read.reference_end+1] += 1

    return coverage

def bam_list_to_h5(bam_list, bed, output='result.h5'):

    bed = pd.read_csv(bed, header=None, index_col=0, sep='\t',
                      names=['contig', 'start', 'end'])

    iterators = [pysam.AlignmentFile(bam, 'rb')
                 for bam in bam_list]

    handle = h5py.File(output, 'w')
    for contig in bed.index:
        coverages = np.zeros((len(iterators), bed.loc['contig', 'end']))
        for i, bam_it in enumerate(iterators):
            coverages[i] = get_contig_coverage(bam_it.fetch(contig))
        handle.write(contig, data=coverages)
            
