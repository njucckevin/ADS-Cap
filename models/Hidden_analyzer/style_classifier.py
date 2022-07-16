import torch
import torch.nn as nn

class Style_Classifier(nn.Module):

    def __init__(self, config):
        super(Style_Classifier, self).__init__()
        self.config = config
        self.latent_dim = config.latent_dim
        self.classifier = nn.Linear(self.latent_dim, config.styles_num, bias=True)

    def forward(self, latent_vec):
        return self.classifier(latent_vec)