from .q_encoder import Q_Encoder
from .p_decoder import P_Decoder
from .resnet_feat_encoder import Resnet_feat_encoder
from .objword_feat_encoder import Objword_feat_encoder
import torch
import torch.nn as nn
import random
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class VAE_Framework(nn.Module):

    def __init__(self, config):
        super(VAE_Framework, self).__init__()

        self.config = config
        self.resnet_feat_encoder = Resnet_feat_encoder(config)
        self.objword_feat_encoder = Objword_feat_encoder(config)
        self.q_encoder = Q_Encoder(config)
        self.p_decoder = P_Decoder(config)
        self.weight_feat_dict = {0: 0, 1: 1}
        self.latent_dim = config.latent_dim

    def encode(self, cap, cap_len, feat_vec):
        return self.q_encoder(cap, cap_len, feat_vec)

    def decode(self, feat_vec, style_label, latent_vec, cap, cap_len):
        return self.p_decoder(feat_vec, style_label, latent_vec, cap, cap_len)

    def generate_latent_vec(self, mu, sigma2):

        latent_vec = torch.randn(mu.size()).to(device)
        latent_vec *= torch.sqrt(torch.exp(sigma2))  # scaled
        latent_vec += mu  # shifted
        return latent_vec

    def latent_mu_sigma2_zero(self, mu, sigma2, mu_prior, sigma2_prior, latent_vec, style_label):
        # 将非风格样本对应的mu、sigma2和latent_vec置为0、0、全0向量，使得对于非风格样本不训练编码器，同时有一个清晰的全0latent_vec
        none_style_dim = [i for i in range(style_label.size(0)) if int(style_label[i]) == 4]
        mu[none_style_dim] = torch.zeros(mu.size(1)).to(device)
        sigma2[none_style_dim] = torch.zeros(sigma2.size(1)).to(device)
        mu_prior[none_style_dim] = torch.zeros(mu_prior.size(1)).to(device)
        sigma2_prior[none_style_dim] = torch.zeros(sigma2_prior.size(1)).to(device)
        latent_vec[none_style_dim] = torch.zeros(latent_vec.size(1)).to(device)
        return mu, sigma2, mu_prior, sigma2_prior, latent_vec

    def forward(self, cap, cap_len, obj, obj_num, cap_style, cap_style_len, image_feat, style_label):

        # 注意，数据集给出了每个样本的object word和resnet特征，但是其中对于风格样本，不能使用其resnet特征；对于coco样本，随机使用object word特征和resnet特征
        obj_vec = self.objword_feat_encoder(obj)
        res_vec = self.resnet_feat_encoder(image_feat)
        samples_style = torch.Tensor([1 if int(item) != 4 else 0 for item in style_label]).to(device)
        random_weight_obj = torch.Tensor([1 if samples_style[i] == 1 else self.weight_feat_dict[random.randint(0, 1)] for i in range(cap.size(0))]).to(device)
        random_weight_res = 1-random_weight_obj
        feat_vec = obj_vec*(random_weight_obj.unsqueeze(1).expand(obj_vec.shape))+res_vec*(random_weight_res.unsqueeze(1).expand(res_vec.shape))

        mu_pri, sigma2_pri = self.q_encoder.generate_prior(feat_vec)
        mu, sigma2 = self.encode(cap_style, cap_style_len, feat_vec)
        # mu, sigma2 = self.encode(cap, cap_len)
        latent_vec = self.generate_latent_vec(mu, sigma2)

        mu, sigma2, mu_pri, sigma2_pri, latent_vec = self.latent_mu_sigma2_zero(mu, sigma2, mu_pri, sigma2_pri, latent_vec, style_label)

        logit = self.decode(feat_vec, style_label, latent_vec, cap[:, :-1], cap_len-1)
        return logit, mu, sigma2, mu_pri, sigma2_pri, latent_vec, obj_vec, res_vec, feat_vec

    def generate_onesample(self, feat_vec, style_label, latent_vec):

        sentence = self.p_decoder.greedy(feat_vec, style_label, latent_vec)
        return sentence

    def generate_beamsearch(self, feat_vec, style_label, latent_vec):

        sentence, sentences = self.p_decoder.beam_search(feat_vec, style_label, latent_vec)
        return sentence, sentences
