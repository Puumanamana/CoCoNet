import re
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

import torch.optim as optim
import torch

from torch_models import CompositionModel, CoverageModel, CoCoNet
from generators import CompositionGenerator, CoverageGenerator

def initialize_model(model_type, config, pretrained_path=None):
    '''
    Build neural network model: either
    composition only, coverage only or both
    '''

    if model_type == 'composition':
        model = CompositionModel(*config.input_shapes["composition"], **config.arch['composition'])

        if pretrained_path is not None:
            checkpoint = torch.load(pretrained_path)
            model.load_state_dict(checkpoint)
            model.train()

    elif model_type == 'coverage':
        model = CoverageModel(*config.input_shapes["coverage"], **config.arch['coverage'])
    else:
        compo_model = initialize_model("composition", config, pretrained_path=pretrained_path)
        cover_model = initialize_model("coverage", config)
        model = CoCoNet(compo_model, cover_model, **config.arch['combination'])

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

def load_data(config, mode='test'):
    '''
    Return data generator by using the parameters
    in the config object
    '''

    if mode == 'test':
        batch_size = get_npy_lines(config.outputs['fragments'][mode])
    else:
        batch_size = config.train['batch_size']

    pairs = config.outputs['fragments']

    generators = []

    if config.model_type != 'coverage':
        generators.append(CompositionGenerator(pairs[mode],
                                               fasta=config.inputs['filtered']['fasta'],
                                               batch_size=batch_size,
                                               kmer_list=config.kmer_list,
                                               rc=config.rc,
                                               norm=config.norm))
    if config.model_type != 'composition':
        generators.append(CoverageGenerator(pairs[mode],
                                            coverage_h5=config.inputs['filtered']['coverage_h5'],
                                            batch_size=batch_size,
                                            load_batch=config.train['load_batch'],
                                            window_size=config.wsize,
                                            window_step=config.wstep))

    if config.model_type not in ['composition', 'coverage']:
        generators = zip(*generators)

    if mode == 'test':
        generators = list(generators)[0]

    return generators

def train(model, config):
    '''
    Train neural network:
    - Generate feature vectors (composition, coverage)
    - Forward pass through network
    - Backward pass and optimization (Adam)
    - Display confusion table and other metrics every 200 batches
    - Single epoch training
    '''

    x_test = load_data(config, mode='test')
    training_generator = load_data(config, mode='train')

    print("Setting labels")
    labels = {mode: get_labels(pairs)
              for mode, pairs in config.outputs['fragments'].items()}

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.train['learning_rate']
    )

    n_train = get_npy_lines(config.outputs['fragments']['train'])
    batch_size = config.train['batch_size']
    n_batches = int(n_train/batch_size)
    running_loss = 0

    print("Training starts")
    for i, batch_x in enumerate(training_generator):
        # zero the parameter gradients
        optimizer.zero_grad()

        truth = labels["train"][i*batch_size:(i+1)*batch_size]

        # forward + backward + optimize
        loss = model.compute_loss(
            model(*batch_x),
            truth
        )
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Get test results
        if (i % 500 == 499) or (i+1 == n_batches):
            outputs_test = test_summary(model,
                                        data={'x': x_test, 'y': labels["test"]},
                                        i=i, n_batches=n_batches)
            # model.eval()
            # outputs_test = model(*x_test)
            # model.train()

            print("\nRunning Loss: {}".format(running_loss))
            # get_confusion_table(outputs_test, labels["test"], done=i/n_batches)

            running_loss = 0

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, config.outputs['net']['model'])

    outputs_test.update({'truth': labels["test"].long()})

    # Save last test performance to file
    pd.DataFrame({
        k: probs.detach().numpy()[:, 0]
        for k, probs in outputs_test.items()
    }).to_csv(config.outputs['net']['test'], index=False)

    print('Finished Training')

def test_summary(model, data=None, config=None, i=1, n_batches=1):
    '''
    Run model on test data and outputs confusion table
    '''
    if data is None:
        data = {
            'x': load_data(config, mode='test'),
            'y': get_labels(config.outputs['fragments']['test'])
        }
    model.eval()
    outputs_test = model(*data['x'])
    model.train()
    get_confusion_table(outputs_test, data["y"], done=i/n_batches)
    return outputs_test

def get_confusion_table(preds, truth, done=0):
    '''
    Confusion table and other metrics for
    predicted labels [pred] and true labels [true]
    '''

    for key, pred in preds.items():
        conf_df = pd.DataFrame(
            confusion_matrix(truth.detach().numpy().astype(int),
                             (pred.detach().numpy()[:, 0] > 0.5).astype(int)),
            columns=["0 (Pred)", "1 (Pred)"],
            index=["0 (True)", "1 (True)"]
        )

        acc = np.trace(conf_df.values)/np.sum(conf_df.values)
        false_pos = conf_df.iloc[0, 1]
        false_neg = conf_df.iloc[1, 0]

        print("\033[1m {:.2%} done -- {}: Accuracy={:.2%} ({} FP, {} FN) - \033[0m"
              .format(done, key, acc, false_pos, false_neg))
