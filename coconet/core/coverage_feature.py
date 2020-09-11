from pathlib import Path

import numpy as np
import h5py
import pysam

from coconet.core.feature import Feature
from coconet.tools import run_if_not_exists


class CoverageFeature(Feature):

    def __init__(self, **kwargs):
        Feature.__init__(self, **kwargs)
        self.ftype = 'coverage'

    def get_contigs(self, key='h5'):
        handle = self.get_handle()
        contigs = list(handle.keys())
        handle.close()

        return np.array(contigs)

    def n_samples(self):
        handle = self.get_handle()
        first_elt = list(handle.keys())[0]
        n_samples = handle[first_elt].shape[0]
        handle.close()

        return n_samples

    @run_if_not_exists()
    def to_h5(self, valid_nucleotides, output=None, logger=None, **filtering):
        if self.path.get('bam', None) is None:
            return
        
        counts = np.zeros(7)
        
        iterators = [pysam.AlignmentFile(bam, 'rb') for bam in self.path['bam']]
        handle = h5py.File(str(output), 'w')

        for k, (contig, positions) in enumerate(valid_nucleotides):
            size = dict(raw=len(positions), filt=sum(positions))
            coverages = np.zeros((len(iterators), size['filt']), dtype='uint32')

            for i, bam_it in enumerate(iterators):
                it = bam_it.fetch(contig, 1, size['raw'])
                
                (cov_i, counts_i) = get_contig_coverage(it, length=size['raw'], **filtering)
                coverages[i] = cov_i[positions]
                counts += counts_i
                
            handle.create_dataset(contig, data=coverages)

            # Report progress
            if logger is not None and k % 1000 == 0 and k > 0:
                logger.debug(f'Coverage: {k:,} contigs processed')

        handle.close()
        self.path['h5'] = Path(output)

        counts[1:] /= counts[0]
        return counts

    def write_singletons(self, output=None, min_prevalence=0, noise_level=0.1):

        with open(output, 'w') as writer:
            header = ['contigs', 'length'] + [f'sample_{i}' for i in range(self.n_samples())]
            writer.write('\t'.join(header))
            h5_handle = self.get_handle()

            for ctg, data in h5_handle.items():
                ctg_coverage = data[:].mean(axis=1)
                prevalence = sum(ctg_coverage > noise_level)

                if prevalence < min_prevalence:
                    info = map(str, [ctg, data.shape[1]] + ctg_coverage.astype(str).tolist())

                    writer.write('\n{}'.format('\t'.join(info)))

            h5_handle.close()


#============ Useful functions for coverage estimation ============#

def get_contig_coverage(iterator, length, **filtering):
    coverage = np.zeros(length, dtype='uint32')

    counts = np.zeros(7)
    for read in iterator:
        conditions = filter_aln(read, **filtering)
        counts[0] += 1
        counts[1] += not read.is_secondary
        counts[2:] += conditions
        
        if all(conditions[2:]):
            # Need to handle overlap between forward and reverse read
            # bam files coordinates are 1-based --> offset
            coverage[read.reference_start-1:read.reference_end] += 1

    return (coverage, counts)

def filter_aln(aln, min_mapq=50, tlen_range=None, min_coverage=0, flag=3852):
    rlen = aln.query_length if aln.query_length > 0 else aln.infer_query_length()
    
    return np.array([
        not aln.is_unmapped,
        aln.mapping_quality >= min_mapq,
        aln.query_alignment_length / rlen >= min_coverage / 100,
        aln.flag & flag == 0,
        (tlen_range is None
         or (tlen_range[0] <= abs(aln.template_length) <= tlen_range[1]))
    ])
