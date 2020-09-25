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


logger = logging.getLogger('learning')

def initialize_model(model_type, input_shapes, architecture):
    """
    Initialize `model_type` coconet model using provided `architecture`.

    Args:
        model_type (string): Either 'composition' or 'coverage'. Anything else will use both
        input_shapes (list or dict): Input shapes for the model. Needs to be a dictionary if both features are used.
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
        compo_model = initialize_model("composition", input_shapes['composition'], architecture['composition'])
        cover_model = initialize_model("coverage", input_shapes['coverage'], architecture['coverage'])
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
            checkpoint = torch.load(config.io['model'])
            model.load_state_dict(checkpoint['state'])
            model.eval()
        except RuntimeError:
            logger.critical(
                ('Could not load network model. '
                 'Is the model checkpoint corrupted?')
            )
        except FileNotFoundError:
            logger.critical(
                ('Could not load network model. '
                 f'File {config.io["model"]} not found')
            )

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

def load_data(fasta=None, coverage=None, pairs=None, mode='test', batch_size=None, **kwargs):
    """
    Setup data generators from the raw data and computed pairs.

    Args:
        fasta (str): path to fasta file
        coverage (str): path to .h5 coverage file
        pairs (str): path to .npy pair file
        mode (str): test or train mode
        batch_size (int): neural network batch size
        kwargs (dict): Additional parameters passed to the generator
    Returns:
        One of the following:
        zip(CompositionGenerator, CoverageGenerator)
        CompositionGenerator
        CoverageGenerator
        list (when `mode` is test)
        
    """

    if mode == 'test':
        batch_size = get_npy_lines(pairs)

    generators = []

    if fasta is not None:
        generators.append(CompositionGenerator(pairs, fasta,
                                               batch_size=batch_size,
                                               kmer=kwargs['kmer'],
                                               rc=kwargs['rc'],
                                               norm=kwargs['norm']))
    if coverage is not None:
        generators.append(CoverageGenerator(pairs, coverage,
                                            batch_size=batch_size,
                                            load_batch=kwargs['load_batch'],
                                            wsize=kwargs['wsize'],
                                            wstep=kwargs['wstep']))

    if len(generators) > 1:
        generators = zip(*generators)
    else:
        generators = generators[0]

    if mode == 'test':
        generators = list(generators)[0]

    return generators

@run_if_not_exists()
def train(model, fasta=None, coverage=None, pairs=None, test_output=None,
          output=None, batch_size=None, **kwargs):
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
        kwargs (dict): Additional learning parameters
    Returns:
        None
    """
    (x_test, x_train_gen) = (
        load_data(fasta=fasta, coverage=coverage, pairs=pairs[mode],
                  mode=mode, batch_size=batch_size, **kwargs)
        for mode in ['test', 'train']
    )
    (y_train, y_test) = (
        get_labels(pairs['train']), get_labels(pairs['test']
        ).detach().numpy().astype(int)[:, 0])

    optimizer = optim.Adam(model.parameters(), lr=kwargs['learning_rate'])
    n_examples = get_npy_lines(pairs['train'])
    n_batches = n_examples // batch_size
    losses_buffer = deque(maxlen=500)

    for i, batch_x in enumerate(x_train_gen, 1):
        optimizer.zero_grad()
        loss = model.compute_loss(model(*batch_x), y_train[(i-1)*batch_size:i*batch_size])
        loss.backward()
        optimizer.step()

        losses_buffer.append(loss.item())

        # Get test results
        if (i % (n_batches//10) == 0) or (i == n_batches):
            predictions = run_test(model, x_test)
            scores = get_test_scores(predictions, y_test)

            train_msg = f"Loss<{np.mean(losses_buffer):.3f}>"
            test_msg = ', '.join(f"{name}<{score['acc']:.1%}>"
                                 for (name, score) in scores.items())
            if logger is not None:
                logger.info((
                    f'Batch #{i:,}/{n_batches:,} - '
                    f'Training loss: {train_msg}, '
                    f'Test accuracy: {test_msg}'
                ))

    torch.save({
        'state': model.state_dict(), 'optimizer': optimizer.state_dict(), 'loss': loss,
    }, output)

    # Save last test performance to file
    predictions.update({'truth': y_test})
    pd.DataFrame(predictions).to_csv(test_output, index=False)

def run_test(model, x_test):
    """
    Run model on test data

    Args:
        model (CompositionModel, CoverageNodel or CoCoNet)
        x_test (list): Test inputs
    Returns
        dict: predictions from model for each feature
    """

    model.eval()
    outputs_test = model(*x_test)
    model.train()

    return {key: val.detach().numpy()[:, 0] for (key, val) in outputs_test.items()}

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

    for key, pred in preds.items():
        pred_bin = (np.array(pred) > 0.5).astype(int)

        conf_df = pd.DataFrame(
            confusion_matrix(truth, pred_bin, labels=[0, 1]),
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
def save_repr_all(model, fasta=None, coverage=None, output=None,
                  n_frags=30, frag_len=1024,
                  rc=True, kmer=4, wsize=64, wstep=32):
    """
    - Calculate intermediate representation for all fragments of all contigs
    - Save it in a .h5 file

    Args:
        model (CompositionModel, CoverageNodel or CoCoNet)
        fasta (str): path to fasta file    
        coverage (str): path to .h5 coverage file
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

    repr_h5 = {key: h5py.File(filename, 'w') for key, filename in output.items()}

    for contig in SeqIO.parse(fasta, "fasta"):
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
                np.apply_along_axis(
                    lambda x: avg_window(x, wsize, wstep), 2, coverage_genome
                ).astype(np.float32))

            feature_arrays.append(x_coverage)

        x_repr = model.compute_repr(*feature_arrays)

        for key, handle in repr_h5.items():
            handle.create_dataset(contig.id, data=x_repr[key].detach().numpy(), dtype=np.float32)

    for handle in repr_h5.values():
        handle.close()
