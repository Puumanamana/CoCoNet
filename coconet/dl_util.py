"""
Functions to train the neural networks
"""

from collections import deque
import re
import logging

import h5py
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from Bio import SeqIO

import torch.optim as optim
import torch

from coconet.core.torch_models import CompositionModel, CoverageModel, CoCoNet
from coconet.core.generators import CompositionGenerator, CoverageGenerator
from coconet.tools import run_if_not_exists, get_kmer_frequency, avg_window


logger = logging.getLogger('<learning>')

def initialize_model(model_type, input_shapes, architecture):
    """
    Initialize `model_type` coconet model using provided `architecture`.

    Args:
        model_type (string): Either 'composition' or 'coverage'. Anything else will use both
        input_shapes (list or dict): Input shapes for the model.
          Needs to be a dictionary if both features are used.
        architecture (dict): Network architecture for each model type
    Returns:
        CompositionModel, CoverageNodel or CoCoNet
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_type == 'composition':
        model = CompositionModel(input_shapes, **architecture)

    elif model_type == 'coverage':
        model = CoverageModel(*input_shapes, **architecture)

    else:
        compo_model = initialize_model("composition",
                                       input_shapes['composition'],
                                       architecture['composition'])
        cover_model = initialize_model("coverage",
                                       input_shapes['coverage'],
                                       architecture['coverage'])
        model = CoCoNet(compo_model, cover_model, **architecture['merge'])

    model.to(device)

    return model

def load_model(config, from_checkpoint=False):
    """
    Wrapper around initialize_model. Loads model with parameters in config
    and loads weights if from_checkpoint is set to True.

    Args:
        config (coconet.core.Configuration object)
        from_checkpoint (bool): whether to load a pre-trained model
    Returns:
        CompositionModel, CoverageNodel or CoCoNet
    """

    input_shapes = config.get_input_shapes()
    architecture = config.get_architecture()
    model = initialize_model('-'.join(config.features), input_shapes, architecture)

    if from_checkpoint:
        try:
            checkpoint = torch.load(str(config.io['model']))
            model.load_state_dict(checkpoint['state'])
            model.eval()
        except RuntimeError as err:
            logger.critical(
                ('Could not load network model. '
                 'Is the model checkpoint corrupted?')
            )
            raise RuntimeError from err
        except FileNotFoundError as err:
            logger.critical(
                ('Could not load network model. '
                 f'File {config.io["model"]} not found')
            )
            raise FileNotFoundError from err

    return model

def get_labels(pairs_file):
    """
    Extract label from pair file
    = 1 if both species are identical,
    = 0 otherwise

    Args:
        pairs_file (str): npy file with structured numpy array
    Returns:
        torch.Tensor: Binary tensor of size (n_pairs, 1)
    """

    ctg_names = np.load(pairs_file)['sp']
    labels = (ctg_names[:, 0] == ctg_names[:, 1]).astype(np.float32)[:, None]

    return torch.from_numpy(labels)

def get_npy_lines(filename):
    """
    Count the number of lines in a .npy file by parsing the header

    Args:
        filename (str): path to npy file
    Returns:
        int: line count
    """

    with open(filename, 'rb') as handle:
        handle.read(10) # Skip the binary part in header
        try:
            header = handle.readline().decode()
            n_lines = int(re.findall(r'\((\d+), \d+\)', header)[0])
        except IndexError:
            print("Failed to parse npy header")
            n_lines = np.load(filename).shape[0]

    return n_lines

@run_if_not_exists()
def train(model, fasta=None, coverage=None, pairs=None, test_output=None,
          output=None, test_batch=500, patience=5, load_batch=200, learning_rate=1e-3,
          batch_size=256, kmer=4, rc=True, wsize=64, wstep=32, threads=1):
    """
    Train neural network:
    - Generate feature vectors (composition, coverage)
    - Forward pass through network
    - Backward pass and optimization (Adam)
    - Display confusion table and other metrics
    - Single epoch training

    Args:
        model (CompositionModel, CoverageNodel or CoCoNet)
        fasta (str): path to fasta file
        coverage (str): path to .h5 coverage file
        pairs (str): path to .npy pair file
        test_output (str): filename to save neural network test results
        output (str): filename to save neural network model
        batch_size (int): Mini-batch for learning
        test_batch (int): Test network every {test_batch} mini-batches
        patience (int): Stop training if test accuracy does not improve for {patience}*{test_batch}
        batch_size (int): neural network batch size
        load_batch (int): number of training batches to load simulaneously
        learning_rate (float): learning rate for AdamOptimizer
        kmer (int): kmer size for composition vector
        rc (bool): whether to add the reverse complement to the composition count (canonical kmers)
        wsize (int): window size for coverage smoothing
        wstep (int): subsampling step for coverage smoothing
        threads (int): number of threads to use
    Returns:
        None
    """

    x_train = dict()
    x_test = dict()
    if fasta is not None:
        x_test['composition'] = next(CompositionGenerator(
            pairs['test'], fasta, batch_size=0,
            kmer=kmer, rc=rc, threads=threads
        ))
        x_train['composition'] = CompositionGenerator(
            pairs['train'], fasta, batch_size=batch_size,
            kmer=kmer, rc=rc, threads=threads
        )
    if coverage is not None:
        x_test['coverage']=next(CoverageGenerator(
            pairs['test'], coverage, batch_size=0,
            load_batch=load_batch, wsize=wsize, wstep=wstep
        ))
        x_train['coverage'] = CoverageGenerator(
            pairs['train'], coverage, batch_size=batch_size,
            load_batch=load_batch, wsize=wsize, wstep=wstep
        )

    if len(x_train) == 1:
        x_train = next(iter(x_train.values()))
        x_test = next(iter(x_test.values()))
    else:
        x_train = zip(*x_train.values())
        x_test = list(x_test.values())

    (y_train, y_test) = (get_labels(pairs['train']), get_labels(pairs['test']))

    # Training start
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    n_examples = get_npy_lines(pairs['train'])
    n_batches = n_examples // batch_size
    test_scores = deque(maxlen=patience)

    for i, batch_x in enumerate(x_train, 1):
        optimizer.zero_grad()
        loss = model.compute_loss(model(*batch_x), y_train[(i-1)*batch_size:i*batch_size])
        loss.backward()
        optimizer.step()

        if (i % test_batch != 0) and (i != n_batches):
            continue

        # Get test results and save if improvements
        metrics = run_test(model, x_test, y_test, test_scores, output, test_output)

        test_acc = ', '.join(f'{name}<{scores["acc"]:.1%}>' for (name, scores) in metrics.items())
        logger.info(f'Test accuracy after {i:,} batches (max={n_batches:,}): {test_acc}')

        # Stop if there are no significant improvement for {patience} test batches
        if len(test_scores) == patience and np.min(test_scores) == test_scores[0]:
            logger.info('Early stopping')
            return

def run_test(model, x, y, test_scores, model_output=None, test_output=None):
    """
    Args:
        model (CompositionModel, CoverageNodel or CoCoNet)
        x (list of torch.Tensor): input test features
        y (torch.Tensor): output test features
        test_scores (Queue): previous test results
        test_output (str): filename to save neural network test results
        model_output (str): filename to save neural network model
    """
    model.eval()
    pred = model(*x)
    model.train()

    scores = get_test_scores(pred, y)

    if 'combined' in scores:
        # score = scores['combined']['acc']
        score = model.loss_op(pred['combined'], y).mean().item()
    else:
        # score = next(iter(scores.values()))['acc']
        score = model.loss_op(next(iter(pred.values())), y).mean().item()

    test_scores.append(score)

    # Save model if best so far
    if score <= min(test_scores):
        torch.save(dict(state=model.state_dict()), model_output)

        pred.update(dict(truth=y))
        pred = pd.DataFrame({key: vec.detach().numpy()[:, 0] for (key, vec) in pred.items()})
        pred.to_csv(test_output, index=False)

    return scores


def get_test_scores(preds, truth):
    """
    Confusion table and other metrics for
    predicted labels [pred] and true labels [true]

    Args:
        preds (dict): model prediction for each feature
        truth (np.array or list): true labels
    Returns:
        dict: scores for each feature and each metric
    """
    scores = {}

    y_true = truth.detach().numpy()[:, 0]
    for key, pred in preds.items():
        pred_bin = (np.array(pred.detach().numpy()[:, 0]) > 0.5).astype(int)

        conf_df = pd.DataFrame(
            confusion_matrix(y_true, pred_bin, labels=[0, 1]),
            columns=["0 (Pred)", "1 (Pred)"],
            index=["0 (True)", "1 (True)"]
        )

        scores[key] = {
            'acc': np.trace(conf_df.values)/np.sum(conf_df.values),
            'fp': conf_df.iloc[0, 1],
            'fn': conf_df.iloc[1, 0]
        }

    return scores

@run_if_not_exists()
def save_repr_all(model, fasta=None, coverage=None, dtr=None, output=None,
                  n_frags=30, frag_len=1024, min_ctg_len=2048,
                  rc=True, kmer=4, wsize=64, wstep=32):
    """
    - Calculate intermediate representation for all fragments of all contigs
    - Save it in a .h5 file

    Args:
        model (CompositionModel, CoverageNodel or CoCoNet)
        fasta (str): path to fasta file
        coverage (str): path to .h5 coverage file
        dtr (str): path to DTR contig list (to exclude)
        output (dict): filename to save latent representations for each feature
        n_frags (int): number of equal size fragments to split contigs
        frag_len (int): size of fragments
        rc (bool): whether to take the reverse complements of kmer composition
        kmer (int): kmer for composition feature. Must be the same as the one used
          for the training.
        wsize (int): window size for coverage smoothing. Must be the same as the
          one used for the training.
        wstep (int): window step for coverage smoothing. Must be the same as the
          one used for the training.
    Returns:
        None
    """

    if 'coverage' in output:
        cov_h5 = h5py.File(coverage, 'r')

    dtr_contigs = set()
    if dtr is not None and dtr.is_file():
        dtr_contigs |= set(ctg.split('\t')[0].strip() for ctg in open(dtr))

    repr_h5 = {key: h5py.File(filename, 'w') for key, filename in output.items()}

    for contig in SeqIO.parse(fasta, "fasta"):
        if contig.id in dtr_contigs or len(contig.seq) < min_ctg_len:
            continue

        step = int((len(contig)-frag_len) / n_frags)

        fragment_boundaries = [(step*i, step*i+frag_len) for i in range(n_frags)]

        feature_arrays = []

        if 'composition' in repr_h5:
            x_composition = torch.from_numpy(np.stack([
                get_kmer_frequency(str(contig.seq)[start:stop], kmer=kmer, rc=rc)
                for (start, stop) in fragment_boundaries
            ]).astype(np.float32)) # Shape = (n_frags, 4**k)

            feature_arrays.append(x_composition)

        if 'coverage' in repr_h5:
            fragment_slices = np.array([np.arange(start, stop)
                                        for (start, stop) in fragment_boundaries])
            coverage_genome = np.array(cov_h5[contig.id][:]).astype(np.float32)[:, fragment_slices]
            coverage_genome = np.swapaxes(coverage_genome, 1, 0)

            x_coverage = torch.from_numpy(
                avg_window(coverage_genome, wsize, wstep, axis=2).astype(np.float32)
            )

            feature_arrays.append(x_coverage)

        x_repr = model.compute_repr(*feature_arrays)

        for key, handle in repr_h5.items():
            handle.create_dataset(contig.id, data=x_repr[key].detach().numpy(), dtype=np.float32)

    for handle in repr_h5.values():
        handle.close()
