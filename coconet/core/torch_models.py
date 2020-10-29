"""
Definition of pytorch models for CoCoNet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class CompositionModel(nn.Module):
    """
    Submodel for composition feature
    """

    def __init__(self, input_size, neurons=None):
        super().__init__()

        self.compo_shared = nn.Linear(input_size, neurons[0])
        self.compo_siam = nn.Linear(2*neurons[0], neurons[1])
        self.compo_prob = nn.Linear(neurons[1], 1)

        self.loss_op = nn.BCELoss(reduction='none')

    def compute_repr(self, x):
        """
        Representation of a composition input
        (Before merging the 2 inputs)
        """
        return dict(composition=F.relu(self.compo_shared(x)))

    def combine_repr(self, *x):
        """
        Combine representation given 2 composition vectors
        Siamese network to make output symetrical
        """
        x_siam1 = F.relu(self.compo_siam(torch.cat(x, axis=1)))
        x_siam2 = F.relu(self.compo_siam(torch.cat(x[::-1], axis=1)))
        x = torch.max(x_siam1, x_siam2)

        return x

    def get_coconet_input(self, *x):
        """
        Run the network up to last representation.
        It will be used by the CoCoNet model
        """

        x = [self.compute_repr(xi)['composition'] for xi in x]
        x = self.combine_repr(*x)

        return x

    def forward(self, *x):
        """
        Run the composition model on its own
        """

        x = self.get_coconet_input(*x)
        x = torch.sigmoid(self.compo_prob(x))

        return dict(composition=x)

    def compute_loss(self, pred, truth):
        return self.loss_op(pred["composition"], truth).mean()

class CoverageModel(nn.Module):
    """
    Submodel for coverage feature
    """

    def __init__(self, input_size, n_samples, neurons=None,
                 n_filters=64, kernel_size=16, conv_stride=8):
        super().__init__()

        self.conv_layer = nn.Conv1d(n_samples, n_filters, kernel_size, conv_stride)
        conv_out_dim = (n_filters,
                        (input_size-kernel_size)//conv_stride + 1)
        self.cover_shared = nn.Linear(np.prod(conv_out_dim), neurons[0])
        self.cover_siam = nn.Linear(2*neurons[0], neurons[1])
        self.cover_prob = nn.Linear(neurons[1], 1)

        self.loss_op = nn.BCELoss(reduction='none')

    def compute_repr(self, x):
        """
        Representation of a coverage input
        (Before merging the 2 inputs)
        """

        x = F.relu(self.conv_layer(x))
        x = F.relu(self.cover_shared(x.view(x.shape[0], -1)))
        return dict(coverage=x)

    def combine_repr(self, *x):
        """
        Combine representation given 2 coverage vectors
        Siamese network to make output symetrical
        """

        x_siam1 = F.relu(self.cover_siam(torch.cat(x, axis=1)))
        x_siam2 = F.relu(self.cover_siam(torch.cat(x[::-1], axis=1)))

        x = torch.max(x_siam1, x_siam2)

        return x

    def get_coconet_input(self, *x):
        """
        Run the network up to last representation.
        It will be used by the CoCoNet model
        """

        x = [self.compute_repr(xi)['coverage'] for xi in x]
        x = self.combine_repr(*x)
        return x

    def forward(self, *x):
        """
        Run the coverage model on its own
        """

        x = self.get_coconet_input(*x)
        x = torch.sigmoid(self.cover_prob(x))

        return dict(coverage=x)

    def compute_loss(self, pred, truth):
        return self.loss_op(pred["coverage"], truth).mean()

class CoCoNet(nn.Module):
    """
    Combined model for CoCoNet
    """

    def __init__(self, composition_model, coverage_model, neurons=32):
        super().__init__()
        self.composition_model = composition_model
        self.coverage_model = coverage_model

        self.dense = nn.Linear(composition_model.compo_prob.in_features
                               + coverage_model.cover_prob.in_features,
                               neurons)
        self.prob = nn.Linear(neurons, 1)

        self.loss_op = nn.BCELoss(reduction='none')

    def compute_repr(self, x1, x2):
        """
        Compute representation for both sub-models
        """

        latent_repr = self.composition_model.compute_repr(x1)
        latent_repr.update(self.coverage_model.compute_repr(x2))

        return latent_repr

    def combine_repr(self, *latent_repr):
        """
        Combine representation for both sub-models
        Compute the final probability
        """

        compo_repr = self.composition_model.combine_repr(
            latent_repr[0]["composition"], latent_repr[1]["composition"]
        )
        cover_repr = self.coverage_model.combine_repr(
            latent_repr[0]["coverage"], latent_repr[1]["coverage"]
        )
        combined = F.relu(self.dense(
            torch.cat([compo_repr, cover_repr], axis=1)
        ))

        return torch.sigmoid(self.prob(combined))

    def forward(self, x1, x2):
        """
        Compute probabilities of all 3 networks
        (Composition, Coverage, Combined)
        """

        compo_repr = self.composition_model.get_coconet_input(*x1)
        cover_repr = self.coverage_model.get_coconet_input(*x2)

        combined = F.relu(self.dense(torch.cat([compo_repr, cover_repr], axis=1)))

        compo_prob = torch.sigmoid(self.composition_model.compo_prob(compo_repr))
        cover_prob = torch.sigmoid(self.coverage_model.cover_prob(cover_repr))
        combined_prob = torch.sigmoid(self.prob(combined))

        x = dict(composition=compo_prob, coverage=cover_prob, combined=combined_prob)
        return x

    def compute_loss(self, pred, truth):
        """
        Get all 3 losses
        """

        loss_compo = self.loss_op(pred['composition'], truth)
        loss_cover = self.loss_op(pred['coverage'], truth)
        loss_combined = self.loss_op(pred['combined'], truth)

        losses = loss_compo+loss_cover+2*loss_combined
        return losses.mean()
