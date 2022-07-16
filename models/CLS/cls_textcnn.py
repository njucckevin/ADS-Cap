import torch
import torch.nn as nn
import pickle
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TextCNN(nn.Module):

    def __init__(self, config):
        super(TextCNN, self).__init__()

        self.config = config
        with open(config.vocab, 'rb') as f:
            vocab = pickle.load(f)
        self.vocab_size = vocab.get_size()
        self.embed_dim = config.embed_dim_c
        self.embed = nn.Embedding(self.vocab_size, self.embed_dim)

        self.num_filters = [20, 20, 20, 20, 20]
        self.filter_sizes = [1, 2, 3, 4, 5]
        self.feature_dim = sum(self.num_filters)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, n, (f, self.embed_dim)) for (n, f) in zip(self.num_filters, self.filter_sizes)
        ])
        self.highway = nn.Linear(self.feature_dim, self.feature_dim)

        self.classifier = weight_norm(nn.Linear(self.feature_dim, 2, bias=True))
        self.dropout = nn.Dropout(0.5)

    def forward(self, cap):
        """
        get sentences features
        :param result: (batch_size, cap_len)
        :return: features
        """
        embeddings = self.embed(cap)
        features = self.get_feature(embeddings)
        label_pred = self.classifier(self.dropout(features))
        return label_pred

    def get_feature(self, inp):
        """提取一个句子的特征
        inp: batch_size * max_seq_len
        return: batch_size * feature_dim
        """
        emb = inp.unsqueeze(1)  # batch_size * 1 * max_seq_len * embed_dim
        convs = [F.relu(conv(emb)).squeeze(3) for conv in self.convs]  # [batch_size * num_filter * length]
        pools = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in convs]  # [batch_size * num_filter]
        pred = torch.cat(pools, 1)  # tensor: batch_size * feature_dim
        highway = self.highway(pred)
        features = torch.sigmoid(highway) * F.relu(highway) + (1. - torch.sigmoid(highway)) * pred  # highway
        return features