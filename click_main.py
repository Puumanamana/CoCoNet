'''
Root script to run CoCoNet

Inputs:
Outputs:
'''

import click

from Bio import SeqIO
import numpy as np
import torch

from parser_info import make_decorator
from tools import check_inputs, get_outputs, get_input_shapes, get_architecture
from preprocessing import format_assembly, bam_list_to_h5, filter_h5
from fragmentation import make_pairs
from dl_util import initialize_model, train, save_repr_all
from clustering import fill_adjacency_matrix, iterate_clustering

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

@click.group(context_settings=CONTEXT_SETTINGS)
def main():
    '''
    Greeter
    '''

    print('\n#### Starting CoCoNet ####\n')

@main.command(help='Preprocess the contig coverage')
@make_decorator('io', 'general', 'preproc')
def preprocessing(fasta, coverage, output, **kw):
    '''
    Preprocessing steps
    '''

    check_inputs(fasta, coverage)
    path = get_outputs(fasta, output, **kw)

    if coverage[0].suffix == '.bam':
        if not path['filt_fasta'].is_file():
            format_assembly(fasta, path['filt_fasta'], min_length=kw['min_ctg_len'])
        if not path['filt_h5'].is_file():
            bam_list_to_h5(
                fasta=path['filt_fasta'], coverage_bam=coverage, output=path['filt_h5'], singleton_file=path['singleton'],
                threads=kw['threads'], min_prevalence=kw['min_prevalence'], tmp_dir=kw['tmp_dir'],
                min_qual=kw['min_mapping_quality'], flag=kw['flag'], fl_range=kw['fl_range']
            )

    # Input is h5 formatted
    if not (path['filt_fasta'].is_file() and path['filt_h5'].is_file()):
        filter_h5(fasta, coverage[0], path['filt_fasta'], path['filt_h5'],
                  singleton_file=path['singleton'],
                  min_length=kw['min_ctg_len'], min_prevalence=kw['min_prevalence'])

@main.command(help='Fragmentation tool to split the contigs')
@make_decorator('io', 'general', 'frag')
def fragmentation(fasta, coverage, output, **kw):
    '''
    Fragmentation steps
    '''

    check_inputs(fasta, coverage)
    path = get_outputs(fasta, output, **kw)

    assembly = [contig for contig in SeqIO.parse(path['filt_fasta'], "fasta")]

    assembly_idx = {'test': np.random.choice(len(assembly), int(0.1*len(assembly)))}
    assembly_idx['train'] = np.setdiff1d(range(len(assembly)), assembly_idx['test'])

    n_examples = {'train': kw['n_train'], 'test': kw['n_test']}

    for mode, pair_file in path['pairs'].items():
        print("Making {} pairs".format(mode))
        make_pairs([assembly[idx] for idx in assembly_idx[mode]],
                   kw['fragment_step'],
                   kw['fragment_length'],
                   output=pair_file,
                   n_examples=n_examples[mode])

@main.command(help='Train neural network')
@make_decorator('io', 'general', 'dl')
def learn(fasta, coverage, output, **kw):
    '''
    Deep learning model
    '''

    check_inputs(fasta, coverage)
    path = get_outputs(fasta, output, **kw)

    torch.set_num_threads(kw['threads'])

    input_shapes = get_input_shapes(kw['kmer'], kw['fragment_length'], not kw['no_rc'],
                                    path['filt_h5'], kw['wsize'], kw['wstep'])
    architecture = get_architecture(**kw)

    model = initialize_model('CoCoNet', input_shapes, architecture)

    pos_args = [path[name] for name in ['filt_fasta', 'filt_h5', 'pairs', 'nn_test']]
    train(model, *pos_args, output=path['model'], **kw)

    checkpoint = torch.load(path['model'])
    model.load_state_dict(checkpoint['state'])
    model.eval()

    save_repr_all(model, path['filt_fasta'], path['filt_h5'],
                  output=path['repr'],
                  n_frags=kw['n_frags'],
                  frag_len=kw['fragment_length'],
                  rc=not kw['no_rc'],
                  wsize=kw['wsize'], wstep=kw['wstep'])
    return model

@main.command(help='Cluster contigs')
@make_decorator('io', 'general', 'cluster')
@click.pass_context
def cluster(context, fasta, coverage, output, **kw):
    '''
    Make adjacency matrix and cluster contigs
    '''

    check_inputs(fasta, coverage)
    path = get_outputs(fasta, output, **kw)

    model = context.forward(learn)

    fill_adjacency_matrix(model, path['repr'], path['adjacency_matrix'],
                          n_frags=kw['n_frags'], max_neighbors=kw['max_neighbors'])

    if not path['refined_assignments'].is_file():
        iterate_clustering(model, path, **kw)

@main.command(help='Run complete pipeline')
@make_decorator('io', 'general', 'preproc', 'frag', 'dl', 'cluster')
@click.pass_context
def run(context, **kwargs):
    '''
    Run complete pipeline
    '''

    context.forward(preprocessing)
    context.forward(fragmentation)
    context.forward(learn)
    context.forward(cluster)

if __name__ == '__main__':
    main()
