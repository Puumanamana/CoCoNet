import subprocess
from pathlib import Path
import csv

import numpy as np
import h5py

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

    def filter_bams(self, outdir=None, **kwargs):
        if 'bam' not in self.path:
            return

        self.path['filt_bam'] = []
        for bam in self.path['bam']:
            output = Path(outdir, f'{bam.stem}_filtered.bam')
            filter_bam_aln(bam, output=output, **kwargs)
            self.path['filt_bam'].append(output)
    
    @run_if_not_exists()
    def bam_to_tsv(self, output=None):
        if 'filt_bam' not in self.path:
            return

        cmd = (['samtools', 'depth', '-d', '20000']
               + [bam.resolve() for bam in self.path['filt_bam']])

        with open(str(output), 'w') as outfile:
            subprocess.call(cmd, stdout=outfile)

        self.path['tsv'] = Path(output)
    
    @run_if_not_exists()
    def tsv_to_h5(self, valid_nucl_pos, output=None):
        if 'tsv' not in self.path:
            return
        
        writer = h5py.File(output, 'w')

        # Save the coverage of each contig in a separate file
        current_ctg = None
        depth_buffer = None

        with open(self.path['tsv'], 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='\t')

            for entry in csv_reader:
                (ctg, pos) = entry[:2]
            
                # 1st case: contig is not in the assembly (filtered out in previous step)
                ctg_len = 1 + valid_nucl_pos.get(ctg, [-1])[-1]

                if ctg_len == 0:
                    continue

                # 2nd case: it's a new contig in the depth file
                if ctg != current_ctg:
                    # If it's not the first one, we save it
                    if current_ctg is not None:
                        pos_subset = valid_nucl_pos[current_ctg]
                        writer.create_dataset(current_ctg, data=depth_buffer[:, pos_subset])
                    # Update current contigs
                    current_ctg = ctg
                    depth_buffer = np.zeros((len(entry)-2, ctg_len), dtype=np.uint32)

                # Fill the contig with depth info
                depth_buffer[:, int(pos)-1] = [int(x) for x in entry[2:]]

        pos_subset = valid_nucl_pos[current_ctg]
        writer.create_dataset(current_ctg, data=depth_buffer[:, pos_subset])
        writer.close()

        self.path['h5'] = Path(output)

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
                    
@run_if_not_exists()
def filter_bam_aln(bam, bed=None, min_qual=0, flag=0, fl_range=None, output=None, threads=1):
    '''
    Run samtools view to filter quality
    '''

    cmds = [
        ['samtools', 'view', '-h', '-L', bed.resolve(), '-@', str(threads),
         '-q', str(min_qual), '-F', str(flag), bam.resolve()],
        ['samtools', 'sort', '-@', str(threads), '-o', output],
        ['samtools', 'index', '-@', str(threads), output]
    ]

    if fl_range:
        cmd = ['awk', '($9=="") | ({} < sqrt($9^2) < {})'.format(*fl_range)]
        cmds.insert(1, cmd)

    processes = []
    for i, cmd in enumerate(cmds[:-1]):
        stdin = None
        stdout = None

        if 0 < i < len(cmds)-1:
            stdin = processes[-1].stdout
        if i < len(cmds)-1:
            stdout = subprocess.PIPE

        process = subprocess.Popen(cmd, stdout=stdout, stdin=stdin)
        processes.append(process)

    processes[-2].wait()
    processes[-1].wait()
