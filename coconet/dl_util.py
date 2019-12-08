from collections import deque
import re

import h5py
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from Bio import SeqIO

import torch.optim as optim
import torch

from coconet.torch_models import CompositionModel, CoverageModel, CoCoNet
from coconet.generators import CompositionGenerator, CoverageGenerator
from coconet.tools import run_if_not_exists, get_kmer_frequency, avg_window

def initialize_model(model_type, input_shapes, architecture):
    '''
    Build neural network model: either
    composition only, coverage only or both
    '''

    if model_type == 'composition':
        neurons = architecture.pop('neurons')
        model = CompositionModel(input_shapes, *neurons, **architecture)

    elif model_type == 'coverage':
        neurons = architecture.pop('neurons')
        model = CoverageModel(*input_shapes, *neurons, **architecture)

    else:
        compo_model = initialize_model("composition", input_shapes['composition'], architecture['composition'])
        cover_model = initialize_model("coverage", input_shapes['coverage'], architecture['coverage'])
        model = CoCoNet(compo_model, cover_model, **architecture['merge'])

    return model

def load_model(config):
    '''
    Load model:
    - initiliaze model with parameters in config
    - load weights with file defined in config
    '''

    input_shapes = config.get_input_shapes()
    architecture = config.get_architecture()
    model = initialize_model('CoCoNet', input_shapes, architecture)

    checkpoint = torch.load(config.io['model'])
    model.load_state_dict(checkpoint['state'])
    model.eval()

    return model

def get_labels(pairs_file):
    '''
    Extract label from pair file
    = 1 if both species are identical,
    = 0 otherwise
    '''

    ctg_names = np.load(pairs_file)['sp']
    labels = (ctg_names[:, 0] == ctg_names[:, 1]).astype(np.float32)[:, None]

    return torch.from_numpy(labels)

def get_npy_lines(filename):
    '''
    Count #lines in a .npy file
    by parsing the header
    '''

    with open(filename, 'rb') as handle:
        handle.read(10) # Skip the binary part in header
        try:
            header = handle.readline().decode()
            n_lines = int(re.findall(r'\((\d+), \d+\)', header)[0])
        except IndexError:
            print("Failed to parse header")
            n_lines = np.load(filename).shape[0]

    return n_lines

def load_data(fasta, h5, pairs, mode='test', model_type='CoCoNet', batch_size=None, **args):
    '''
    Return data generator by using the parameters
    in the config object
    '''

    if mode == 'test':
        batch_size = get_npy_lines(pairs)

    generators = []

    if model_type != 'coverage':
        generators.append(CompositionGenerator(pairs, fasta,
                                               batch_size=batch_size,
                                               kmer=args['kmer'],
                                               rc=args['rc'],
                                               norm=args['norm']))
    if model_type != 'composition':
        generators.append(CoverageGenerator(pairs, h5,
                                            batch_size=batch_size,
                                            load_batch=args['load_batch'],
                                            window_size=args['wsize'],
                                            window_step=args['wstep']))

    if model_type not in ['composition', 'coverage']:
        generators = zip(*generators)

    if mode == 'test':
        generators = list(generators)[0]

    return generators

@run_if_not_exists()
def train(model, fasta, coverage, pairs, nn_test_path, output=None, batch_size=None, **args):
    '''
    Train neural network:
    - Generate feature vectors (composition, coverage)
    - Forward pass through network
    - Backward pass and optimization (Adam)
    - Display confusion table and other metrics
    - Single epoch training
    '''

    (x_test, x_train_gen) = (load_data(fasta, coverage, pairs[mode], mode=mode, batch_size=batch_size, **args)
                             for mode in ['test', 'train'])
    (y_train, y_test) = (get_labels(pairs['train']), get_labels(pairs['test']))

    optimizer = optim.Adam(model.parameters(), lr=args['learning_rate'])
    n_batches = get_npy_lines(pairs['train']) // batch_size
    losses_buffer = deque(maxlen=500)

    for i, batch_x in enumerate(x_train_gen, 1):
        optimizer.zero_grad()
        loss = model.compute_loss(model(*batch_x), y_train[(i-1)*batch_size:i*batch_size])
        loss.backward()
        optimizer.step()

        losses_buffer.append(loss.item())

        # Get test results
        if (i % (1 + n_batches//5) == 0) or (i == n_batches):
            scores = test_summary(model, i/n_batches, data={'x': x_test, 'y': y_test})
            print("\nRunning Loss: {}".format(np.mean(losses_buffer)))

    torch.save({
        'state': model.state_dict(), 'optimizer': optimizer.state_dict(), 'loss': loss,
    }, output)

    # Save last test performance to file
    scores.update({'truth': y_test.long().numpy()[:, 0]})
    pd.DataFrame(scores).to_csv(nn_test_path, index=False)

    print('Finished Training')

def test_summary(model, progress, data=None, **args):
    '''
    Run model on test data and outputs confusion table
    '''
    if data is None:
        pos_args = [args.pop(k) for k in ['filt_fasta', 'filt_coverage', 'pairs']]
        data = {
            'x': load_data(*pos_args, mode='test', **args),
            'y': get_labels(pos_args[-1])
        }
    model.eval()

    outputs_test = model(*data['x'])
    model.train()
    get_confusion_table(outputs_test, data["y"], done=progress)
    return {key: val.detach().numpy()[:, 0] for (key, val) in outputs_test.items()}

def get_confusion_table(preds, truth, done=0):
    '''
    Confusion table and other metrics for
    predicted labels [pred] and true labels [true]
    '''
    for key, pred in preds.items():
        conf_df = pd.DataFrame(
            confusion_matrix(truth.detach().numpy().astype(int)[:, 0],
                             (pred.detach().numpy()[:, 0] > 0.5).astype(int)),
            columns=["0 (Pred)", "1 (Pred)"],
            index=["0 (True)", "1 (True)"]
        )

        acc = np.trace(conf_df.values)/np.sum(conf_df.values)
        false_pos = conf_df.iloc[0, 1]
        false_neg = conf_df.iloc[1, 0]

        print("\033[1m {:.1%} done -- {}: Accuracy={:.2%} ({} FP, {} FN) - \033[0m"
              .format(done, key, acc, false_pos, false_neg))

@run_if_not_exists()
def save_repr_all(model, fasta, coverage, n_frags=30, frag_len=1024, output=None, rc=True, kmer=4, wsize=64, wstep=32):
    '''
    - Calculate intermediate representation for all fragments of all contigs
    - Save it in a .h5 file
    '''

    print('Computing intermediate representation of composition and coverage features')

    cov_h5 = h5py.File(coverage, 'r')

    repr_h5 = {key: h5py.File(filename, 'w') for key, filename in output.items()}

    for contig in SeqIO.parse(fasta, "fasta"):
        step = int((len(contig)-frag_len) / n_frags)

        fragment_boundaries = [(step*i, step*i+frag_len) for i in range(n_frags)]

        x_composition = torch.from_numpy(np.stack([
            get_kmer_frequency(str(contig.seq)[start:stop], kmer=kmer, rc=rc)
            for (start, stop) in fragment_boundaries
        ]).astype(np.float32)) # Shape = (n_frags, 4**k)

        fragment_slices = np.array([np.arange(start, stop)
                                    for (start, stop) in fragment_boundaries])
        coverage_genome = np.array(cov_h5.get(contig.id)[:]).astype(np.float32)[:, fragment_slices]
        coverage_genome = np.swapaxes(coverage_genome, 1, 0)

        x_coverage = torch.from_numpy(
            np.apply_along_axis(
                lambda x: avg_window(x, wsize, wstep), 2, coverage_genome
            ).astype(np.float32))

        x_repr = model.compute_repr(x_composition, x_coverage)

        for key, handle in repr_h5.items():
            handle.create_dataset(contig.id, data=x_repr[key].detach().numpy(), dtype=np.float32)

    for handle in repr_h5.values():
        handle.close()
