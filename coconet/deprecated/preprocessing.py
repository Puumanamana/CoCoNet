'''
Preprocessing scripts:

- format_assembly() filters out sequences from a fasta that are too short
from a fasta file
- filter_bam_aln() filters a bam file than don't meet quality, flag
or fragment length criteria
- bam_to_h5() extracts coverage info from bams file and
write to h5 file
- filter_h5() filters out too short sequences from a h5_file

'''

from pathlib import Path
from tempfile import mkdtemp
import shutil
import subprocess
import csv

import h5py
import numpy as np

from Bio import SeqIO
from Bio.Seq import Seq

from coconet.tools import run_if_not_exists

@run_if_not_exists()
def format_assembly(fasta, bed_output='regions.bed', output=None, min_length=2048):
    '''
    Remove N nucleotide from assembly
    Filter out sequences shorter than [min_length]
    Contig info in bed file for bam subsetting
    '''

    if output is None:
        output = '{}_gt{}{}'.format(fasta.stem, min_length, fasta.suffix)

    formated_assembly = []
    bed_data = []

    for contig in SeqIO.parse(fasta, 'fasta'):
        ctg_len = len(contig)
        contig.seq = Seq(str(contig.seq).replace('N', '').upper())

        if len(contig.seq) >= min_length:
            formated_assembly.append(contig)
            bed_data += '\t'.join([contig.id, '1', str(1+ctg_len)]) + '\n'

    SeqIO.write(formated_assembly, output, 'fasta')

    # Contig information in bed file for bam subsetting
    with open(bed_output, 'w') as bedfile:
        bedfile.writelines(bed_data)

    return Path(output)

def filter_bam_aln(bam, bed, threads=1, min_qual=0, flag=0, fl_range=None, outdir=None):
    '''
    Run samtools view to filter quality
    '''

    sorted_output = Path(outdir, f'{bam.stem}_filtered.bam')
    if sorted_output.is_file():
        return sorted_output

    cmds = [
        ['samtools', 'view', '-h', '-L', bed.resolve(), '-@', str(threads),
         '-q', str(min_qual), '-F', str(flag), bam.resolve()],
        ['samtools', 'sort', '-@', str(threads), '-o', sorted_output],
        ['samtools', 'index', '-@', str(threads), sorted_output]
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

    return sorted_output


def bam_to_h5(bam_list, valid_nucl_pos, tmp_dir='/tmp', output='coverage.h5'):
    '''
    - Run samtools depth on all bam file and save it in tmp_dir
    - Read the output and save the result in a h5 file with keys as contigs
    '''

    # Compute
    coverage_tsv = Path(tmp_dir, 'coverage.tsv')

    cmd = (['samtools', 'depth', '-d', '20000']
           + [bam.resolve() for bam in bam_list])

    with open(str(coverage_tsv), 'w') as outfile:
        subprocess.call(cmd, stdout=outfile)
    
    writer = h5py.File(output, 'w')

    # Save the coverage of each contig in a separate file
    current_ctg = None
    depth_buffer = None

    with open(coverage_tsv, 'r') as csv_file:
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
    
    return output

@run_if_not_exists()
def process_bam_coverage(
        bams, assembly, bed, output=None, tmp_dir='auto',
        threads=1, min_prevalence=2, rm_filt_bam=False,
        **bam_filter_params):
    '''
    - Extract the coverage of the sequences in fasta from the bam files
    - Remove N nucleotides from the FASTA and
      remove the corresponding entries from coverage
    - Filter out the contigs with less than [min_prevalence] samples
      with a mean coverage less than 1
    '''


    if tmp_dir == 'auto':
        tmp_dir = mkdtemp()
    else:
        tmp_dir.mkdir(exist_ok=True)

    ctg_seq = {seq.id: seq for seq in SeqIO.parse(assembly, 'fasta')}

    valid_nucl = {
        key: np.array([i for i, letter in enumerate(ctg.seq) if letter.upper() != 'N'])
        for key, ctg in ctg_seq.items()
    }

    filtered_bam_files = [
        filter_bam_aln(bam, bed, threads=threads, outdir=output.parent,
                       **bam_filter_params)
        for bam in sorted(bams)
    ]

    depth_h5 = bam_to_h5(filtered_bam_files, valid_nucl, tmp_dir=tmp_dir, output=output)

    if rm_filt_bam:
        for bam in filtered_bam_files:
            bam.unlink()

    shutil.rmtree(tmp_dir)

    return depth_h5

@run_if_not_exists()
def remove_singletons(coverage_h5, fasta,
                      output='singletons.txt',
                      min_prevalence=0):

    handle = h5py.File(coverage_h5, 'r')
    ctg_ids = list(handle.keys())
    n_samples = handle[ctg_ids[0]].shape[0]
    
    singletons_handle = open(output, 'w')
    singletons_handle.write(
        '\t'.join(['contigs', 'length']+[f'sample_{i}' for i in range(n_samples)])
    )

    contigs = {contig.id: contig for contig in SeqIO.parse(fasta, 'fasta')}

    for ctg_id in ctg_ids:
        # Filter out contig with coverage on only 1 sample
        ctg_coverage = handle[ctg_id][:].mean(axis=1)
        prevalence = sum(ctg_coverage >= 0.1)

        if prevalence < min_prevalence:
            info = map(str, [ctg_id, handle[ctg_id].shape[1]] + ctg_coverage.astype(str).tolist())
            singletons_handle.write('\n{}'.format('\t'.join(info)))
            contigs.pop(ctg_id)

    SeqIO.write(list(contigs.values()), fasta, 'fasta')

def summarize_filtering(fasta_in, fasta_out, singletons=None):
    n_before = sum(1 for line in open(fasta_in) if line.startswith('>'))
    n_after = sum(1 for line in open(fasta_out) if line.startswith('>'))

    n_singletons = -1
    if singletons is not None and Path(singletons).is_file():
        n_singletons = sum(1 for _ in open(singletons))
    
    return f'before: {n_before:,}, after: {n_after:,}, singletons: {n_singletons:,}'
    
