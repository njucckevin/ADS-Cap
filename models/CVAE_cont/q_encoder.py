import torch
import torch.nn as nn
import pickle
from torch.nn.utils.weight_norm import weight_norm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Q_Encoder(nn.Module):
    """生成表示要描述对象的obj_vec（词向量均值）和表示风格的隐变量均值和方差"""
    def __init__(self, config):
        super(Q_Encoder, self).__init__()

        self.config = config
        with open(config.vocab, 'rb') as f:
            vocab = pickle.load(f)
        self.vocab_size = vocab.get_size()
        self.embed_dim = config.embed_dim
        self.hidden_dim = config.hidden_dim
        self.latent_dim = config.latent_dim

        self.embed = nn.Embedding(self.vocab_size, self.embed_dim)
        self.lstmcell = nn.LSTMCell(self.embed_dim, self.hidden_dim)

        self.feat2mu = weight_norm(nn.Linear(self.config.align_dim, self.latent_dim))

        self.hidden2mu = weight_norm(nn.Linear(self.hidden_dim, self.latent_dim))
        self.hidden2sigma2 = weight_norm(nn.Linear(self.hidden_dim, self.latent_dim))

        self.init_h = weight_norm(nn.Linear(config.align_dim, self.hidden_dim))
        self.init_c = weight_norm(nn.Linear(config.align_dim, self.hidden_dim))

        self.init_weight()

    def init_weight(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)

    def feat_init_state(self, feat_vec):
        h = self.init_h(feat_vec)
        c = self.init_c(feat_vec)
        return h, c

    def generate_prior(self, feat_vec):
        # 先验分布均值和条件有关，方差设为固定值
        mu = self.feat2mu(feat_vec)
        sigma2 = torch.zeros(mu.shape).to(device)

        return mu, sigma2

    def generate_latent(self, cap, cap_len, feat_vec):

        batch_size = cap.size(0)
        h, c = self.feat_init_state(feat_vec)
        # h = torch.zeros(batch_size, self.hidden_dim).to(device)
        # c = torch.zeros(batch_size, self.hidden_dim).to(device)
        embeddings = self.embed(cap)
        hidden_state = torch.zeros(batch_size, max(cap_len), self.hidden_dim).to(device)

        for t in range(max(cap_len)):
            h, c = self.lstmcell(embeddings[:, t, :], (h, c))
            hidden_state[:, t, :] = h

        hidden_state_last = torch.zeros(batch_size, self.hidden_dim).to(device)
        for i in range(batch_size):
            hidden_state_last[i, :] = hidden_state[i, cap_len[i]-1, :]

        return self.hidden2mu(hidden_state_last), self.hidden2sigma2(hidden_state_last)

    def forward(self, cap, cap_len, feat_vec):

        mu, sigma2 = self.generate_latent(cap, cap_len, feat_vec)
        return mu, sigma2
