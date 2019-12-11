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

    runner = CliRunner()
    fasta = '{}/sim_data/assembly.fasta'.format(LOCAL_DIR)
    bam = ['{}/sim_data/sample_{}.bam'.format(LOCAL_DIR, i) for i in range(1, 3)]
    outdir = Path('./out_test')

    options = (
        ['--output', outdir]
        + ['--n-train', 64]
        + ['--n-test', 8]
        + ['--batch-size', 2]
        + ['--min-prevalence', 0]
        + ['--test-ratio', 0.2]
        + ['--threads', 1]
        + ['--n-frags', 5]
        + ['--compo-neurons', 8, 4]
        + ['--cover-neurons', 8, 4]
        + ['--cover-kernel', 2]
        + ['--wsize', 2]
        + ['--wstep', 2]
    )

    res = runner.invoke(main, ['run', fasta] + bam + options)

    print(res.output)
    print(res.exc_info)

    for filepath in outdir.glob('./*'):
        filepath.unlink()
    outdir.rmdir()

    assert res.exit_code == 0

if __name__ == '__main__':
    test_all()
