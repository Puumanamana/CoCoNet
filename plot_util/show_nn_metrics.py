import re
from glob import glob
import pandas as pd
import torch

import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append('..')

from nn_training import initialize_model, get_npy_lines, get_labels, get_confusion_table
from generators import CompositionGenerator, CoverageGenerator

kmer_list = [2,3,4]
rc = False
fl = 1024
ws = 16

## NN model
model_type = 'CoCoNet'

nn_arch = {
    'composition': { 'neurons': [128,64,32] },
    'coverage': { 'neurons': [128,64,32],
                  'n_filters': 64, 'kernel_size': 7,'conv_stride': 3,
                  'pool_size': 2, 'pool_stride': 1},
    'combination': { 'neurons': [32] }
}

def load_network(input_shapes,model_output):
    model = initialize_model(model_type,input_shapes,
                             composition_args=nn_arch["composition"],
                             coverage_args=nn_arch["coverage"],
                             combination_args=nn_arch["combination"]
    )

    checkpoint = torch.load(model_output)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model

def load_data(dataset):
    ## Input data
    pairs_test = "../output_data/{}/pairs_test.npy".format(dataset)
    n_test = get_npy_lines(pairs_test)
    print("{} test examples to process".format(n_test))

    fasta = "../input_data/{}/assembly_gt2048.fasta".format(dataset)
    coverage_h5 = glob("../input_data/{}/coverage_*.h5".format(dataset))[0]

    X_test = [
        next(CompositionGenerator(fasta,pairs_test,batch_size=n_test,
                                  kmer_list=kmer_list, rc=rc)),
        next(CoverageGenerator(coverage_h5,pairs_test,
                               batch_size=n_test, load_batch=1, window_size=ws))
    ]

    truth = get_labels(pairs_test)

    print('Data loaded')

    return X_test,truth

def plot_data(pred,truth):
    data = pd.DataFrame({k: v.detach().numpy()[:,0] for k,v in pred.items()})
    data['truth'] = truth.numpy()[:,0]
    
    sns.pairplot(data, hue='truth',vars=['composition','coverage','combined'], diag_kind='hist',
                 plot_kws={'s': 5, 'alpha': 0.1, 'edgecolor': ''},
                 diag_kws={'bins': 50})
    plt.show()

    return data

def main(dataset='sim_5'):
    if len(sys.argv) > 1:
        dataset = sys.argv[1]
    model_output = "../output_data/{}/{}.pth".format(dataset,model_type)

    try:
        if 'viral' in dataset:
            n_samples = 17
        else:
            n_samples = int(re.findall("[A-z_]+(\d+)",dataset)[0])
    except:
        n_samples = 3
    print(dataset,n_samples)
    
    input_shapes = {
        'composition': [ sum([int(4**k / (1+rc)) for k in kmer_list ]) ],
        'coverage': (int(fl/ws), n_samples)
    }
    
    model = load_network(input_shapes,model_output)
    X_test, truth = load_data(dataset)
    
    pred = model(*X_test)

    get_confusion_table(pred,truth)

    data = plot_data(pred,truth)

    return data

if __name__ == '__main__':
    data = main()
