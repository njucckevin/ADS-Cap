import torch
import torch.nn as nn
import pickle
from torch.nn.utils.weight_norm import weight_norm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Resnet_feat_encoder(nn.Module):

    def __init__(self, config):
        super(Resnet_feat_encoder, self).__init__()

        self.config = config

        self.resnet_feat_dim = config.image_dim
        self.align_dim = config.align_dim

        self.resnet2align = weight_norm(nn.Linear(self.resnet_feat_dim, self.align_dim))

    def forward(self, resnet_feat):
        return self.resnet2align(resnet_feat)
