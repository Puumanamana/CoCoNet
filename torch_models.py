import torch
import torch.nn as nn
import torch.nn.functional as F

class CompositionModel(nn.Module):
    def __init__(self, input_size, neurons):
        super(CompositionModel, self).__init__()
        self.compo_shared = nn.Linear(input_size,neurons[0])
        self.compo_dense = nn.Linear(2*neurons[0],neurons[1])

    def compute_repr(self,x):
        return F.relu(self.compo_shared(x))
        
    def forward(self, *x):
        x = [ self.compute_repr(xi) for xi in x ]
        x = F.relu( self.compo_dense(torch.cat(x)) )
        
        return x

class CoverageModel(nn.Module):
    def __init__(self, input_size, n_samples, neurons,
                 n_filters=64, kernel_size=16, conv_stride=8,
                 pool_size=4, pool_stride=2):
        super(CoverageModel, self).__init__()

        self.conv_layer = nn.Conv1d(n_samples,n_filters,kernel_size,stride)
        conv_out_dim = (n_filters,n_samples,
                        int((input_size-kernel_size)/stride)+1)
        self.pool = nn.MaxPool1d(pool_size,pool_stride)
        pool_out_dim = (n_filters,n_samples,
                        int((conv_out_dim[-1]-pool_size)/pool_stride)+1)
        self.cover_shared = nn.Linear(np.prod(pool_out_dim), neurons[0])
        self.cover_dense = nn.Linear(2*neurons[0], neurons[1])

    def compute_repr(self,x):
        x = F.relu(self.conv_layer(x))
        x = F.relu(self.pool(x))
        x = F.relu(self.cover_shared(x.view(x.shape[0],-1)))
        return x
        
    def forward(self, x):
        x = [ self.compute_repr(xi) for xi in x ]
        x = F.relu( self.cover_dense(torch.cat(x)) )
        return x

class CoCoNet(nn.Module):
    def __init__(self, composition_model, coverage_model, neurons):
        super(CoCoNet, self).__init__()
        self.composition_model = composition_model
        self.coverage_model = coverage_model

        self.compo_prob = nn.Linear(neurons[0],1)
        self.cover_prob = nn.Linear(neurons[0],1)

        self.dense = nn.Linear(2*neurons[0],neurons[1])
        self.prob = nn.Linear(neurons[1],neurons[2])

    
    def forward(self, x1, x2):
        
        return x
    
