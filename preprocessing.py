"""
Preprocessing scripts:

- format_assembly() filters out sequences from a fasta that are too short
from a fasta file
- filter_bam_aln() filters a bam file than don't meet quality, flag
or fragment length criteria
- bam_to_h5() extracts coverage info from a single bam file and
write to h5 file
- bam_list_to_h5() converts a list of bam files to h5, applies a
prevalence filter and remove sequences that are too short
- filter_h5() filters out too short sequences from a h5_file

"""

import shutil
from os.path import basename, splitext, exists
from tempfile import mkdtemp
import csv
import subprocess

import h5py
import numpy as np

from Bio import SeqIO
from Bio.Seq import Seq

from progressbar import progressbar

def format_assembly(fasta, output=None, min_length=2048):
    '''
    Remove N nucleotide from assembly and
    filter out sequences shorter than [min_length]
    '''

    if output is None:
        base, ext = splitext(fasta)
        output = "{}_gt{}{}".format(base, min_length, ext)

    formated_assembly = [genome for genome in SeqIO.parse(fasta, "fasta")
                         if len(str(genome.seq).replace('N','')) >= min_length]

    SeqIO.write(formated_assembly, output, "fasta")

def filter_bam_aln(bam, threads, min_qual, flag, fl_range):
    '''
    Run samtools view to filter quality
    '''

    sorted_output = "{}_q{}-F{}_fl{}-{}_sorted.bam"\
        .format(splitext(bam)[0], min_qual, flag, *fl_range)

    if exists(sorted_output):
        return sorted_output

    cmds = [
        ['samtools', 'view', '-h', bam, '-@', str(threads)],
        ['awk', '($9>{} && $9<{}) || ($9<-{} && $9>-{}) || ($9=="")'.format(*(2*fl_range))],
        ['samtools', 'view', '-bh', bam, '-@', str(threads), '-q', str(min_qual), '-F', str(flag)],
        ['samtools', 'sort', '-@', str(threads), '-o', sorted_output],
        ['samtools', 'index', '-@', str(threads), sorted_output]
    ]

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

def bam_to_h5(bam, temp_dir, ctg_info):
    '''
    - Run samtools depth on bam file and save it in temp_dir
    - Read the output and save the result in a h5 file with keys as contigs
    '''

    outputs = {fmt: "{}/{}.{}".format(temp_dir, splitext(basename(bam))[0],fmt)
               for fmt in ['txt', 'h5']}

    with open(outputs['txt'], "w") as outfile:
        subprocess.call(["samtools", "depth", "-d", "20000", bam],
                        stdout=outfile)

    n_entries = sum(1 for _ in open(outputs['txt']))
    h5_handle = h5py.File(outputs['h5'], 'w')

    # Save the coverage of each contig in a separate file
    current_ctg = None
    depth_buffer = None

    with open(outputs['txt'], 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')

        for ctg, pos, d_i in progressbar(csv_reader, max_value=n_entries):
            # 1st case: contig is not in the assembly (filtered out in previous step)
            ctg_len = ctg_info.get(ctg, None)

            if ctg_len is None:
                continue

            # 2nd case: it's a new contig in the depth file
            if ctg != current_ctg:
                # If it's not the first one, we save it
                if current_ctg is not None:
                    h5_handle.create_dataset("{}".format(current_ctg),
                                             data=depth_buffer)
                # Update current contigs
                current_ctg = ctg
                depth_buffer = np.zeros(ctg_len, dtype=np.uint32)

            # Fill the contig with depth info
            depth_buffer[int(pos)-1] = int(d_i)

    return outputs['h5']

def bam_list_to_h5(fasta, coverage_bam, output, threads=1, min_prevalence=2, **bam_filter_params):
    '''
    - Extract the coverage of the sequences in fasta from the bam files
    - Remove N nucleotides from the FASTA and
      remove the corresponding entries from coverage
    - Filter out the contigs with less than [min_prevalence] samples
      with a mean coverage less than 1
    '''

    temp_dir = mkdtemp()
    ctg_info = {seq.id: len(seq.seq)
                for seq in SeqIO.parse(fasta, "fasta")}

    filtered_bam_files = [filter_bam_aln(bam, threads, **bam_filter_params)
                          for bam in sorted(coverage_bam)]

    depth_h5_files = [bam_to_h5(bam, temp_dir, ctg_info)
                      for bam in filtered_bam_files]

    # Collect everything in a [N_samples,genome_size] matrix
    coverage_h5 = h5py.File(output, 'w')

    ctg_seq = {seq.id: seq for seq in SeqIO.parse(fasta, "fasta")}
    assembly_no_n = []

    h5_handles = [h5py.File(f, 'r') for f in depth_h5_files]

    for ctg in ctg_info:
        ctg_coverage = [h.get(ctg) for h in h5_handles]
        ctg_coverage = np.vstack([x[:] if x is not None
                                  else np.zeros(ctg_info[ctg], dtype=np.uint32)
                                  for x in ctg_coverage])

        # Take care of the N problem
        loc_acgt = np.array([i for i, letter in enumerate(ctg_seq[ctg]) if letter != 'N'])

        ctg_coverage = ctg_coverage[:, loc_acgt]

        # Filter out contig with coverage on only 1 sample
        sample_coverage = ctg_coverage.sum(axis=1)
        if sum(sample_coverage >= 0.1*ctg_info[ctg]) <= min_prevalence:
            continue

        coverage_h5.create_dataset(ctg, data=ctg_coverage)

        # Process the sequence
        seq_no_n = ctg_seq[ctg]
        seq_no_n.seq = Seq(str(seq_no_n.seq).replace('N', '').upper())

        assembly_no_n.append(seq_no_n)

    SeqIO.write(assembly_no_n, fasta, "fasta")

    # Remove temp directory
    shutil.rmtree(temp_dir)

def filter_h5(inputs, min_length=2048, min_prevalence=2):
    '''
    Filter coverage h5 and only keep sequences
    longer than [min_length]
    '''

    h5_reader = h5py.File(inputs['raw']['coverage_h5'], 'r')
    h5_writer = h5py.File(inputs['filtered']['coverage_h5'], 'w')
    assembly = {seq.id: seq for seq in SeqIO.parse(inputs['raw']['fasta'], 'fasta')}

    for ctg in h5_reader:
        data = h5_reader.get(ctg)[:]
        prevalence = sum(data.sum(axis=1) > 0.1*data.shape[1])
        if (data.shape[1] >= min_length) and (prevalence >= min_prevalence):
            h5_writer.create_dataset(ctg, data=data[:])
        else:
            del assembly[ctg]

    h5_reader.close()
    h5_writer.close()

    SeqIO.write(list(assembly.values()), inputs['filtered']['fasta'], 'fasta')

    # Zip the raw coverage to save space
    subprocess.call(['gzip', inputs['raw']['coverage_h5']])
