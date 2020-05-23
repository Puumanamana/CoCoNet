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

from pathlib import Path
import shutil
from tempfile import mkdtemp
import csv
import subprocess

from tqdm import tqdm
import h5py
import numpy as np

from Bio import SeqIO
from Bio.Seq import Seq

from coconet.tools import run_if_not_exists

@run_if_not_exists()
def format_assembly(fasta, output=None, min_length=2048):
    '''
    Remove N nucleotide from assembly and
    filter out sequences shorter than [min_length]
    '''

    if output is None:
        output = "{}_gt{}{}".format(fasta.stem, min_length, fasta.suffix)

    formated_assembly = [genome for genome in SeqIO.parse(fasta, "fasta")
                         if len(str(genome.seq).replace('N', '')) >= min_length]

    SeqIO.write(formated_assembly, output, "fasta")

    return Path(output)

def filter_bam_aln(bam, threads, min_qual, flag, fl_range, outdir=None):
    '''
    Run samtools view to filter quality
    '''

    sorted_output = Path("{}/{}_q{}-F{}_fl{}_sorted.bam".format(
        outdir, bam.stem, min_qual, flag, ''.join(map(str, fl_range)))
    )
    if sorted_output.is_file():
        return sorted_output

    cmds = [['samtools', 'view', '-bh', bam.resolve(), '-@', str(threads)]]

    if flag is not None:
        cmds.append(['samtools', 'view', '-h', '-@', str(threads),
                     '-q', str(min_qual), '-F', str(flag)])

    if fl_range:
        cmds.append([
            'awk', 
            '($9>{} && $9<{}) || ($9<-{} && $9>-{}) || ($9=="")'
            .format(*(2*fl_range))])

    cmds += [
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

def bam_to_h5(bam, tmp_dir, ctg_info):
    '''
    - Run samtools depth on bam file and save it in tmp_dir
    - Read the output and save the result in a h5 file with keys as contigs
    '''

    outputs = {fmt: Path("{}/{}.{}".format(tmp_dir, bam.stem, fmt))
               for fmt in ['txt', 'h5']}

    with open(outputs['txt'], "w") as outfile:
        subprocess.call(["samtools", "depth", "-d", "20000", bam.resolve()],
                        stdout=outfile)

    n_lines = sum(1 for _ in open(outputs['txt']))

    h5_handle = h5py.File(outputs['h5'], 'w')

    # Save the coverage of each contig in a separate file
    current_ctg = None
    depth_buffer = None

    print('Converting bam to hdf5')
    with open(outputs['txt'], 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')

        for ctg, pos, d_i in tqdm(csv_reader, total=n_lines):
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

@run_if_not_exists()
def bam_list_to_h5(fasta, coverage_bam, output=None,
                   threads=1, min_prevalence=2, singleton_file='./singletons.txt',
                   tmp_dir='auto', rm_filt_bam=False, **bam_filter_params):
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

    ctg_info = {seq.id: len(seq.seq)
                for seq in SeqIO.parse(fasta, "fasta")}

    filtered_bam_files = [filter_bam_aln(bam, threads, outdir=output.parent, **bam_filter_params)
                          for bam in sorted(coverage_bam)]

    depth_h5_files = [bam_to_h5(bam, tmp_dir, ctg_info)
                      for bam in filtered_bam_files]

    if rm_filt_bam:
        for bam in filtered_bam_files:
            bam.unlink()

    singletons_handle = open(singleton_file, 'w')
    singletons_handle.write(
        '\t'.join(["contigs", "length"]+["sample_{}".format(i) for i in range(len(depth_h5_files))])
    )

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
        coverage_h5.create_dataset(ctg, data=ctg_coverage)

        # Filter out contig with coverage on only 1 sample
        sample_coverage = ctg_coverage.sum(axis=1)
        prevalence = sum(sample_coverage >= 0.1*len(loc_acgt))

        if prevalence < min_prevalence:
            info = map(str, [ctg, len(loc_acgt)] + sample_coverage.astype(str).tolist())
            singletons_handle.write("\n{}".format('\t'.join(info)))
            continue

        # Process the sequence
        seq_no_n = ctg_seq[ctg]
        seq_no_n.seq = Seq(str(seq_no_n.seq).replace('N', '').upper())

        assembly_no_n.append(seq_no_n)

    SeqIO.write(assembly_no_n, fasta, "fasta")
    singletons_handle.close()

    # Remove tmp directory
    shutil.rmtree(tmp_dir)

@run_if_not_exists(keys=('filt_fasta', 'filt_h5'))
def filter_h5(fasta, h5, filt_fasta=None, filt_h5=None,
              min_length=2048, min_prevalence=2, singleton_file="./singletons.txt"):
    '''
    Filter coverage h5 and only keep sequences
    longer than [min_length]
    '''

    h5_reader = h5py.File(h5, 'r')
    h5_writer = h5py.File(filt_h5, 'w')

    singletons_handle = open(singleton_file, 'w')
    assembly = {seq.id: seq for seq in SeqIO.parse(fasta, 'fasta') if seq.id in h5_reader}

    n_samples = h5_reader.get(list(h5_reader.keys())[0]).shape[0]

    singletons_handle.write(
        '\t'.join(["contigs", "length"] + ["sample_{}".format(i) for i in range(n_samples)])
    )

    for ctg in h5_reader:
        data = h5_reader.get(ctg)[:]
        sample_coverage = data.sum(axis=1)
        prevalence = sum(sample_coverage >= 0.1*data.shape[1])

        if data.shape[1] >= min_length:
            h5_writer.create_dataset(ctg, data=data[:])

            if prevalence < min_prevalence:
                info = map(str, [ctg, data.shape[1]] + sample_coverage.tolist())
                singletons_handle.write("\n{}".format('\t'.join(info)))

        if (data.shape[1] < min_length) or (prevalence < min_prevalence):
            del assembly[ctg]

    singletons_handle.close()
    h5_reader.close()
    h5_writer.close()

    SeqIO.write(list(assembly.values()), filt_fasta, 'fasta')
