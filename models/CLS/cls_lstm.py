import torch
import torch.nn as nn
import pickle
from torch.nn.utils.weight_norm import weight_norm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Cls_Classifier(nn.Module):

    def __init__(self):
        super(Cls_Classifier, self).__init__()

        with open('./models/CLS/vocab_cls.pkl', 'rb') as f:
            self.vocab = pickle.load(f)
        self.vocab_size = self.vocab.get_size()
        self.embed_dim = 300
        self.hidden_dim = 300

        self.embed = nn.Embedding(self.vocab_size, self.embed_dim)
        self.lstmcell = nn.LSTMCell(self.embed_dim, self.hidden_dim)

        self.classifier = weight_norm(nn.Linear(self.hidden_dim, 2, bias=True))
        self.dropout = nn.Dropout(0.5)

        self.init_weight()

    def init_weight(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, sen_list):

        sen_id, sen_len = self.vocab.tokenList_to_idList(sen_list, 20)
        cap = torch.Tensor(sen_id).long().to(device).unsqueeze(0)
        cap_len = torch.LongTensor([sen_len]).to(device)
        cap_len += 2

        batch_size = cap.size(0)

        h = torch.zeros(batch_size, self.hidden_dim).to(device)
        c = torch.zeros(batch_size, self.hidden_dim).to(device)
        embeddings = self.embed(cap)
        hidden_state = torch.zeros(batch_size, max(cap_len), self.hidden_dim).to(device)

        for t in range(max(cap_len)):
            h, c = self.lstmcell(embeddings[:, t, :], (h, c))
            hidden_state[:, t, :] = h

        hidden_state_last = torch.zeros(batch_size, self.hidden_dim).to(device)
        for i in range(batch_size):
            hidden_state_last[i, :] = hidden_state[i, cap_len[i]-1, :]

        label_pred = self.classifier(self.dropout(hidden_state_last))

        return label_pred

