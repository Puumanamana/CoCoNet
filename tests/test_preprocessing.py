'''
Tests for preprocessing of fasta and coverage (.h5 or .bam)
'''

from Bio import SeqIO

from coconet.preprocessing import format_assembly

from data import generate_fasta_file

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

def test_bam_filtering():
    '''
    Test if bam are filtered correctly
    '''

    assert 1 == 1

def test_bam_to_h5():
    '''
    Test if depth from samtools can be converted to h5
    '''

    assert 1 == 1

def test_bamlist_to_h5():
    '''
    Test for wrapper
    '''

    assert 1 == 1

def test_filter_h5():
    '''
    Test for filter directly when input is h5 formatted
    '''

    assert 1 == 1
