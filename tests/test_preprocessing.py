'''
Tests for preprocessing of fasta and coverage (.h5 or .bam)
'''

from pathlib import Path

import pandas as pd
from Bio import SeqIO

from coconet.core.feature import Feature
from coconet.core.composition_feature import CompositionFeature
from coconet.core.coverage_feature import CoverageFeature

from .data import generate_fasta_file, generate_h5_file

LOCAL_DIR = Path(__file__).parent

def test_feature_obj_is_created():
    f = Feature(name='x')

    assert f.name == 'x'

def test_composition_feature():
    fasta_raw = generate_fasta_file(20, 10, 15)
    
    f = CompositionFeature(path=dict(fasta=fasta_raw))
    f.filter_by_length('filtered.fasta', summary_output='exclude.txt', min_length=12)

    filtered_lengths = [len(seq.seq) for seq in SeqIO.parse(f.path['filt_fasta'], 'fasta')]
    fasta_raw.unlink()
    f.path['filt_fasta'].unlink()
    Path('exclude.txt').unlink()

    assert filtered_lengths == [20, 15]

def test_coverage_feature():
    h5 = generate_h5_file(10, 20, filename='coverage.h5')
    f = CoverageFeature(path={'h5': h5})
    ctg = f.get_contigs()

    found_2_ctg = len(ctg) == 2
    h5.unlink()
    
    assert found_2_ctg
    
def test_bam_to_h5():
    '''
    Test if bamlist is converted to h5
    '''

    fasta = Path(LOCAL_DIR, 'sim_data', 'assembly.fasta')
    compo = CompositionFeature(path=dict(fasta=fasta, filt_fasta=fasta))
    
    bam_list = [Path(LOCAL_DIR, 'sim_data', f'sample_{i}.bam') for i in [1, 2]]
    h5_out = Path('coverage.h5')


    f = CoverageFeature(path=dict(bam=bam_list))
    f.to_h5(compo.get_valid_nucl_pos(), output=h5_out)

    h5_is_created = h5_out.is_file()
    h5_out.unlink()

    assert h5_is_created


def test_remove_singletons():
    '''
    Test if singletons are removed
    '''

    lengths = [60, 100, 80]
    h5_data = generate_h5_file(*lengths, n_samples=3,
                               baselines=[20, 40, 30],
                               empty_samples=[[False]*3, [True, True, False], [False]*3],
                               filename='coverage.h5')
    fasta = generate_fasta_file(*lengths)
    singleton_file = Path('singletons.txt')

    compo = CompositionFeature(path=dict(filt_fasta=fasta))
    cover = CoverageFeature(path=dict(h5=h5_data))

    cover.remove_singletons(output=singleton_file, min_prevalence=2)
    compo.filter_by_ids(output='filt.fasta', ids_file=singleton_file)

    singletons = pd.read_csv('singletons.txt', sep='\t', header=None).values
    n_filt = sum(1 for _ in SeqIO.parse('filt.fasta', 'fasta'))

    for f in [fasta, h5_data, 'singletons.txt', 'filt.fasta']:
        Path(f).unlink()

    assert singletons.shape == (1, 3)
    assert n_filt == 2
        

if __name__ == '__main__':
    test_remove_singletons()
