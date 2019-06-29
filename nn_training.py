import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

import torch.optim as optim
import torch

from torch_models import CompositionModel, CoverageModel, CoCoNet
from generators import CompositionGenerator, CoverageGenerator

from progressbar import progressbar

def initialize_model(model_type, input_shapes, composition_args=None, coverage_args=None, combination_args=None):
    if model_type == 'composition':
        model = CompositionModel(input_shapes["composition"], **composition_args)
    elif model_type == 'coverage':
        model = CoverageModel(*input_shapes["coverage"],**coverage_args)
    else:
        compo_model = initialize_model("composition",input_shapes, composition_args=composition_args)
        cover_model = initialize_model("coverage", input_shapes, coverage_args=composition_args)
        model = CoCoNet(compo_model,cover_model,**combination_args)

    return model

def get_labels(pairs_file):
    pairs = pd.read_csv(pairs_file,index_col=0,header=[0,1])
    labels = (pairs.A.sp == pairs.B.sp).values.astype(np.float32)[:,None]

    return torch.from_numpy(labels)

def train(model, pairs_file, output, fasta=None, coverage_h5=None,
          batch_size=64, kmer=4, window_size=16, load_batch=1000,
          learning_rate=1e-4):

    training_generators = [
        CompositionGenerator(fasta,pairs_file["train"],
                             batch_size=batch_size,k=kmer),
        CoverageGenerator(coverage_h5, pairs_file["train"],
                          batch_size=batch_size,load_batch=load_batch,window_size=window_size)
    ]

    n_test = sum(1 for _ in open(pairs_file["test"]))-2

    print("Loading test data")
    test_data = [
        next(CompositionGenerator(fasta,pairs_file["test"],batch_size=n_test, k=kmer)),
        next(CoverageGenerator(coverage_h5,pairs_file["test"],
                               batch_size=n_test, load_batch=1, window_size=window_size))
    ]

    print("Setting labels")
    labels = {
        "train": get_labels(pairs_file["train"]),
        "test": get_labels(pairs_file["test"])
    }

    optimizer = optim.Adam(list(model.composition_model.parameters())
                           + list(model.coverage_model.parameters())
                           + list(model.parameters()),
                           lr=learning_rate)

    running_loss = 0

    print("Training starts")
    for i, (X_compo, X_cover) in progressbar(enumerate(zip(*training_generators)),
                                             max_value=len(training_generators[0])):

        # zero the parameter gradients
        optimizer.zero_grad()

        truth = labels["train"][i*batch_size:(i+1)*batch_size]

        # forward + backward + optimize
        outputs = model(X_compo,
                        X_cover)
        loss = model.compute_loss(outputs,truth)

        loss = model.compute_loss(outputs, truth)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Get test results
        if i % 200 == 199:
            outputs_test = model(*test_data)
            print("\nRunning Loss: {}".format(running_loss))
            get_confusion_table(outputs_test,labels["test"])
            
            running_loss = 0

    torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
    }, output)

    print('Finished Training')


def get_confusion_table(preds,truth):

    for key,pred in preds.items():
        conf_mat = pd.DataFrame(confusion_matrix(truth.detach().numpy().astype(int),
                                                 pred.detach().numpy()[:,0].astype(int)),
                                columns=["1 (True)","0 (True)"],
                                index=["0 (Pred)","1 (Pred)"]
        )

        print("\033[1mConfusion matrix for {}\033[0m\n{}\n".format(key, conf_mat))
