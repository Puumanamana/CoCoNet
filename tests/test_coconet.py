'''
Tests for main algorithm
'''

from pathlib import Path

from click.testing import CliRunner

from coconet.coconet import main

LOCAL_DIR = Path(__file__).parent

def test_main():
    '''
    Test greeter
    '''

    runner = CliRunner()
    result = runner.invoke(main)

    assert result.exit_code == 0

def test_all():
    '''
    test parser and overall app
    '''

    pass
    # runner = CliRunner()
    # fasta = '{}/sim_data/'.format(LOCAL_DIR)
    # bam = ['{}/sim_data/sample_{}.bam'.format(LOCAL_DIR, i) for i in range(1, 3)]
    # main_runner = runner.invoke(main, ['run']).runner

    # main_runner.invoke('run')
    # main_runner.env = {'fasta': fasta, 'coverage': bam}
    
    # assert result.exit_code == 0

