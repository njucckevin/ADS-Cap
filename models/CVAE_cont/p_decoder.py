import torch
import torch.nn as nn
import pickle
from torch.nn.utils.weight_norm import weight_norm
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class P_Decoder(nn.Module):

    def __init__(self, config):
        super(P_Decoder, self).__init__()

        self.config = config
        with open(config.vocab, 'rb') as f:
            vocab = pickle.load(f)
        self.vocab_size = vocab.get_size()
        self.embed_dim = config.embed_dim
        self.hidden_dim = config.hidden_dim
        self.latent_dim = config.latent_dim
        self.styles_num = config.styles_num

        self.embed = nn.Embedding(self.vocab_size, self.embed_dim)
        self.latent2embed = nn.Linear(self.latent_dim, self.embed_dim, bias=False)
        self.embed_style = nn.Embedding(self.styles_num, self.embed_dim)
        self.lstmcell = nn.LSTMCell(self.embed_dim+self.embed_dim, self.hidden_dim)

        self.init_h = weight_norm(nn.Linear(self.embed_dim, self.hidden_dim))
        self.init_c = weight_norm(nn.Linear(self.embed_dim, self.hidden_dim))

        self.fc = weight_norm(nn.Linear(self.hidden_dim, self.vocab_size))
        self.dropout = nn.Dropout(0.5)

        self.init_weight()

    def init_weight(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def obj_init_state(self, obj_vec):
        h = self.init_h(obj_vec)
        c = self.init_c(obj_vec)
        return h, c

    def forward(self, obj_vec, style_label, latent_vec, cap, cap_len):

        batch_size = obj_vec.size(0)
        h, c = self.obj_init_state(obj_vec)
        embeddings = self.embed(cap)

        style_embedding = self.embed_style(style_label)
        latent_input = self.latent2embed(latent_vec)
        h, c = self.lstmcell(torch.cat([torch.zeros(latent_input.shape).to(device), style_embedding], dim=1), (h, c))
        h, c = self.lstmcell(torch.cat([latent_input, torch.zeros(style_embedding.shape).to(device)], dim=1), (h, c))

        logit = torch.zeros(batch_size, max(cap_len), self.vocab_size).to(device)

        for t in range(max(cap_len)):
            embeddings_input = embeddings[:, t, :]
            if torch.rand(1).item() < self.config.unk_rate:
                embeddings_input = torch.zeros(batch_size).to(device).long()
                embeddings_input = self.embed(embeddings_input)
            embeddings_input = torch.cat([embeddings_input, style_embedding], dim=1)

            h, c = self.lstmcell(embeddings_input, (h, c))
            pred = self.fc(self.dropout(h))
            logit[:, t, :] = pred

        return logit

    def greedy(self, obj_vec, style_label, latent_vec):

        sentences = torch.ones(1, 1).to(device).long()

        h, c = self.obj_init_state(obj_vec)
        style_embedding = self.embed_style(style_label)
        latent_input = self.latent2embed(latent_vec)
        h, c = self.lstmcell(torch.cat([torch.zeros(latent_input.shape).to(device), style_embedding], dim=1), (h, c))
        h, c = self.lstmcell(torch.cat([latent_input, torch.zeros(style_embedding.shape).to(device)], dim=1), (h, c))

        for i in range(self.config.fixed_len+1):
            embedding = sentences[:, i]
            embedding = self.embed(embedding)
            embedding = torch.cat([embedding, style_embedding], dim=1)

            h, c = self.lstmcell(embedding, (h, c))
            pred = self.fc(self.dropout(h))
            probs = F.softmax(pred, 1)
            score, token_id = torch.max(probs, dim=-1)
            sentences = torch.cat([sentences, token_id.unsqueeze(1)], dim=1)

        return sentences[0]

    def beam_search(self, obj_vec, style_label, latent_vec):

        beam_num = self.config.beam_num

        sample = []
        sample_score = []
        live_k = 1
        dead_k = 0

        hyp_samples = [[1]] * live_k
        hyp_scores = torch.zeros(1).to(device)

        h, c = self.obj_init_state(obj_vec)
        style_embedding_init = self.embed_style(style_label)
        latent_input = self.latent2embed(latent_vec)
        h, c = self.lstmcell(torch.cat([torch.zeros(latent_input.shape).to(device), style_embedding_init], dim=1), (h, c))
        h, c = self.lstmcell(torch.cat([latent_input, torch.zeros(style_embedding_init.shape).to(device)], dim=1), (h, c))

        hyp_status = [0]
        hyp_status[0] = (h, c)

        for i in range(self.config.fixed_len+1):
            sen_size = len(hyp_samples)

            embedding = [hyp_samples[j][-1] for j in range(sen_size)]
            embedding = torch.Tensor(embedding).long().to(device)
            embedding = self.embed(embedding)
            style_embedding = style_embedding_init.expand([sen_size, style_embedding_init.shape[1]])
            embedding = torch.cat([embedding, style_embedding], dim=1)

            h_batch = torch.cat([hyp_status[j][0] for j in range(sen_size)], dim=0)
            c_batch = torch.cat([hyp_status[j][1] for j in range(sen_size)], dim=0)

            h, c = self.lstmcell(embedding, (h_batch, c_batch))
            pred = self.fc(self.dropout(h))
            probs = F.softmax(pred, 1)

            can_score = hyp_scores.expand([self.vocab_size, sen_size]).permute(1, 0)
            can_score = can_score-torch.log(probs)
            can_score_flat = can_score.flatten()
            word_ranks = can_score_flat.argsort()[:(beam_num-dead_k)]
            status_indices = torch.floor_divide(word_ranks, self.vocab_size)
            word_indices = word_ranks % self.vocab_size
            after_score = can_score_flat[word_ranks]

            new_hyp_samples = []
            new_hyp_scores = []
            new_hyp_status = []
            live_k = 0
            for idx, [si, wi] in enumerate(zip(status_indices, word_indices)):
                if int(wi) == 2:
                    sample.append(hyp_samples[si]+[int(wi)])
                    sample_score.append((after_score[idx]))
                    dead_k += 1
                else:
                    live_k += 1
                    new_hyp_samples.append(hyp_samples[si]+[int(wi)])
                    new_hyp_scores.append(after_score[idx])
                    new_hyp_status.append((h[si].unsqueeze(0), c[si].unsqueeze(0)))

            hyp_samples = new_hyp_samples
            hyp_scores = torch.Tensor(new_hyp_scores).to(device)
            hyp_status = new_hyp_status

            if live_k < 1:
                break

        for i in range(len(hyp_samples)):
            sample.append(hyp_samples[i])
            sample_score.append(hyp_scores[i])

        alpha = self.config.beam_alpha
        for j in range(len(sample_score)):
            sample_score[j] = sample_score[j]/(pow((5+len(sample[j])), alpha)/pow(5+1, alpha))

        # 新增：将sample中的beam_num个句子按照得分从高到低（log从低到高）排序
        rank = [item[0] for item in sorted(enumerate(sample_score), key=lambda x:x[1])]
        sample_final = [sample[item] for item in rank]

        min_id = sample_score.index(min(sample_score))

        return sample[min_id], sample_final