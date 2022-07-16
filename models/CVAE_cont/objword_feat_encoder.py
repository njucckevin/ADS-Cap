import torch
import torch.nn as nn
import pickle
from torch.nn.utils.weight_norm import weight_norm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Objword_feat_encoder(nn.Module):

    def __init__(self, config):
        super(Objword_feat_encoder, self).__init__()

        self.config = config

        with open(config.vocab, 'rb') as f:
            vocab = pickle.load(f)
        self.vocab_size = vocab.get_size()
        self.embed_dim = config.embed_dim
        self.align_dim = config.align_dim

        self.embed = nn.Embedding(self.vocab_size, self.embed_dim)

        self.objectword2align = weight_norm(nn.Linear(self.embed_dim, self.align_dim))

    def forward(self, obj):
        obj_embeddings = self.embed(obj)
        obj_vec = obj_embeddings.mean(dim=1)
        return self.objectword2align(obj_vec)

