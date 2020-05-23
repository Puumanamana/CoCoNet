'''
Tests for preprocessing of fasta and coverage (.h5 or .bam)
'''

from pathlib import Path

import pandas as pd
import h5py
from Bio import SeqIO

from coconet.preprocessing import format_assembly, filter_h5, filter_bam_aln
from coconet.preprocessing import bam_to_h5, bam_list_to_h5

from .data import generate_fasta_file, generate_h5_file

LOCAL_DIR = Path(__file__).parent

def test_format_assembly():
    '''
    Test fasta filtering
    '''

    fasta_raw = generate_fasta_file(20, 10, 15)
    fasta_filt = format_assembly(fasta_raw, min_length=12)

    filtered_lengths = [len(seq.seq) for seq in SeqIO.parse(fasta_filt, 'fasta')]

    fasta_raw.unlink()
    fasta_filt.unlink()

    assert filtered_lengths == [20, 15]

def test_format_assembly_when_exists():
    '''
    Test fasta filtering is not performed if already exists
    '''

    fasta_raw = generate_fasta_file(20, 10, 15)
    fasta_filt = format_assembly(fasta_raw, min_length=12)
    fasta_filt_twice = format_assembly(fasta_raw, output=fasta_filt, min_length=12)

    filtered_lengths = [len(seq.seq) for seq in SeqIO.parse(fasta_filt, 'fasta')]

    fasta_raw.unlink()
    fasta_filt.unlink()

    assert filtered_lengths == [20, 15]
    assert fasta_filt_twice is None

def test_filter_bam():
    '''
    Test if bam are filtered correctly
    '''

    bam_file = Path("{}/sim_data/sample_1.bam".format(LOCAL_DIR))
    output = filter_bam_aln(bam_file, 5, 50, 3596, [0, 1000], outdir=LOCAL_DIR)
    assert output.is_file()

    output.unlink()

def test_bam_to_h5():
    '''
    Test if depth from samtools can be converted to h5
    '''

    bam_file = Path("{}/sim_data/sample_1.bam".format(LOCAL_DIR))
    ctg_info = {seq.id: len(seq.seq) for seq in
                SeqIO.parse("{}/sim_data/assembly.fasta".format(LOCAL_DIR), 'fasta')}
    h5 = bam_to_h5(bam_file, '/tmp', ctg_info)

    assert h5.is_file()

    h5.unlink()

def test_bamlist_to_h5():
    '''
    Test for wrapper
    '''

    fasta = Path("{}/sim_data/assembly.fasta".format(LOCAL_DIR))
    bam_list = [Path("{}/sim_data/sample_{}.bam".format(LOCAL_DIR, i)) for i in [1, 2]]
    h5_out = Path('coverage.h5')
    singleton_file = Path('singletons.txt')

    fasta_filt = format_assembly(fasta, min_length=512)
    bam_list_to_h5(fasta_filt, bam_list, output=h5_out, singleton_file=singleton_file,
                   min_qual=50, flag=3596, fl_range=[0, 1000], rm_filt_bam=True)

    assert h5_out.is_file() and singleton_file.is_file()

    for f in [h5_out, singleton_file, fasta_filt]:
        f.unlink()

def test_filter_h5():
    '''
    Test for filter directly when input is h5 formatted
    '''

    lengths = [60, 100, 80]
    h5_data = generate_h5_file(*lengths, n_samples=3,
                                     baselines=[20, 40, 30],
                                     empty_samples=[[False]*3, [True, True, False], [False]*3])
    fasta = generate_fasta_file(*lengths)
    filt = ['filt.fasta', 'filt.h5']

    filter_h5(fasta, h5_data, filt[0], filt[1], min_length=70, min_prevalence=2)

    singletons = pd.read_csv('singletons.txt', sep='\t').values
    n_filt = sum(1 for _ in SeqIO.parse(filt[0], 'fasta'))
    h5_filt_handle = h5py.File(filt[1], 'r')
    h5_data_filt = {k: h5_filt_handle.get(k)[:] for k in h5_filt_handle}

    for f in [h5_data, fasta, 'singletons.txt'] + filt:
        Path(f).unlink()

    assert singletons.shape == (1, 2+3)
    assert n_filt == 1
    assert len(h5_data_filt) == 2

if __name__ == '__main__':
    test_filter_bam()
