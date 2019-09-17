import re
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

import torch.optim as optim
import torch

from progressbar import progressbar

from torch_models import CompositionModel, CoverageModel, CoCoNet
from generators import CompositionGenerator, CoverageGenerator

def initialize_model(model_type, config, pretrained_path=None):
    if model_type == 'composition':
        model = CompositionModel(*config.input_shapes["composition"], **config.arch['composition'])

        if pretrained_path is not None:
            checkpoint = torch.load(pretrained_path)
            model.load_state_dict(checkpoint)
            model.train()
            # for param in model.parameters():
            #     param.requires_grad = False
            # model.eval()

    elif model_type == 'coverage':
        model = CoverageModel(*config.input_shapes["coverage"], **config.arch['coverage'])
    else:
        compo_model = initialize_model("composition", config, pretrained_path=pretrained_path)
        cover_model = initialize_model("coverage", config)
        model = CoCoNet(compo_model, cover_model, **config.arch['combination'])

    return model

def get_labels(pairs_file):
    ctg_names = np.load(pairs_file)['sp']
    labels = (ctg_names[:, 0] == ctg_names[:, 1]).astype(np.float32)[:, None]

    return torch.from_numpy(labels)

def get_npy_lines(filename):
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
    if mode == 'test':
        batch_size = get_npy_lines(config.outputs['fragments'][mode])
    else:
        batch_size = config.train['batch_size']

    pairs = config.outputs['fragments']

    composition_generator = CompositionGenerator(pairs[mode],
                                                 fasta=config.inputs['filtered']['fasta'],
                                                 batch_size=batch_size,
                                                 kmer_list=config.kmer_list,
                                                 rc=config.rc,
                                                 norm=config.norm)
    coverage_generator = CoverageGenerator(pairs[mode],
                                           coverage_h5=config.inputs['filtered']['coverage_h5'],
                                           batch_size=batch_size,
                                           load_batch=config.train['load_batch'],
                                           window_size=config.wsize,
                                           window_step=config.wstep)
    return (composition_generator, coverage_generator)

def train(model, config):
    '''
    Train neural network:
    - Generate feature vectors (composition, coverage)
    - Forward pass through network
    - Backward pass and optimization (Adam)
    - Display confusion table and other metrics every 200 batches
    - Single epoch training
    '''

    (composition_gen_test, coverage_gen_test) = load_data(config, mode='test')
    training_generators = load_data(config, mode='train')

    if config.model_type == 'composition':
        x_test = next(composition_gen_test)
        generator = training_generators[0]
    elif config.model_type == "coverage":
        x_test = next(coverage_gen_test)
        generator = training_generators[1]
    else:
        x_test = [next(composition_gen_test), next(coverage_gen_test)]
        generator = zip(*training_generators)

    print("Setting labels")
    labels = {mode: get_labels(pairs)
              for mode, pairs in config.outputs['fragments'].items()}

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.train['learning_rate']
    )

    n_train = get_npy_lines(config.outputs['fragments']['train'])
    batch_size = config.train['batch_size']
    running_loss = 0

    print("Training starts")
    for i, batch_x in progressbar(enumerate(generator),
                                  max_value=int(n_train/batch_size)):

        # zero the parameter gradients
        optimizer.zero_grad()

        truth = labels["train"][i*batch_size:(i+1)*batch_size]

        # forward + backward + optimize
        outputs = model(*batch_x)

        loss = model.compute_loss(outputs, truth)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Get test results
        if (i % 200 == 199) or (i+1 == int(n_train/batch_size)):
            model.eval()
            outputs_test = model(*x_test)
            model.train()

            print("\nRunning Loss: {}".format(running_loss))
            # get_confusion_table(outputs,truth)
            get_confusion_table(outputs_test, labels["test"])

            running_loss = 0

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, config.outputs['net']['model'])

    test_results = pd.DataFrame({k: v.detach().numpy()[:, 0]
                                 for k, v in outputs_test.items()})
    test_results['truth'] = labels["test"].numpy()[:, 0].astype(int)
    test_results.to_csv(config.outputs['net']['test'], index=False)

    print('Finished Training')

def get_confusion_table(preds, truth):
    '''
    Confusion table and other metrics for
    predicted labels [pred] and true labels [true]
    '''

    for key, pred in preds.items():
        conf_mat = pd.DataFrame(
            confusion_matrix(truth.detach().numpy().astype(int),
                             (pred.detach().numpy()[:, 0] > 0.5).astype(int)),
            columns=["0 (Pred)", "1 (Pred)"],
            index=["0 (True)", "1 (True)"]
        )

        print("\033[1mConfusion matrix for {}\033[0m\n{}\n".format(key, conf_mat))

        conf_mat = conf_mat.values
        metrics = [
            'precision: {:.2%}'.format(conf_mat[1, 1] / np.sum(conf_mat[:, 1])),
            'recall: {:.2%}'.format(conf_mat[1, 1] / np.sum(conf_mat[1, :])),
            'FP_rate: {:.2%}'.format(conf_mat[0, 1] / np.sum(conf_mat[:, 1])),
            'accuracy: {:.2%}'.format(np.trace(conf_mat) / np.sum(conf_mat))
        ]

        print("\33[38;5;38m{}\033[0m\n".format(", ".join(metrics)))
