import os.path

import numpy as np
import torch
from torch import nn

from model.stylegan2_generator import build
from sklearn.decomposition import PCA


class MDSE:
    def __init__(self, params):
        super(MDSE).__init__()

        self.stylegan = build('stylegan2_ffhq1024', use_cuda=params.device == 'cuda')
        self.subspace_model = SubspaceModel(params, self.stylegan.num_layers, self.stylegan.w_space_dim)

    def load_checkpoints(self):
        assert os.path.exists('./checkpoints/subspace_model.pth'), 'checkpoints not found'

        self.subspace_model.load_state_dict(torch.load('./checkpoints/subspace_model.pth'))

    def get_subspace_directions(self, attr):
        pca_vec = PCA().fit(self.subspace_model.data_coef[attr].weight.detach().cpu().numpy()).components_
        directions = np.matmul(pca_vec, self.subspace_model.decompose()[attr].detach().cpu().numpy())
        return directions


class SubspaceModel(nn.Module):
    def __init__(self, params, num_layers, w_space_dim):
        super(SubspaceModel, self).__init__()
        self.num_layers = num_layers
        self.w_space_dim = w_space_dim
        self.attr_dim = params.attr_dim
        self.P = nn.Parameter(torch.randn(sum(params.attr_dim.values()), num_layers * w_space_dim))
        self.data_coef = torch.nn.ModuleDict({k: nn.Embedding(params.num_samples, params.attr_dim[k])
                                              for k in params.attr_dim})

    def forward(self, sample_id):
        a = torch.cat([v(sample_id) for v in self.data_coef.values()], dim=1)
        w = torch.matmul(a, self.P)
        return w.view(-1, self.num_layers, self.w_space_dim)

    def decompose(self):
        P = {}
        total_dim = 0
        for attr, dim in self.attr_dim.items():
            P[attr] = self.P[total_dim: total_dim + dim]
            total_dim += dim
        assert total_dim == self.P.shape[0]
        return P
