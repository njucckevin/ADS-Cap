from .p_decoder import P_Decoder
from .resnet_feat_encoder import Resnet_feat_encoder
from .objword_feat_encoder import Objword_feat_encoder
import torch
import torch.nn as nn
import random
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class EncDec_Framework(nn.Module):

    def __init__(self, config):
        super(EncDec_Framework, self).__init__()

        self.config = config
        self.resnet_feat_encoder = Resnet_feat_encoder(config)
        self.objword_feat_encoder = Objword_feat_encoder(config)
        self.p_decoder = P_Decoder(config)
        self.weight_feat_dict = {0: 0, 1: 1}

    def decode(self, feat_vec, style_label, cap, cap_len):
        return self.p_decoder(feat_vec, style_label, cap, cap_len)

    def forward(self, cap, cap_len, obj, obj_num, image_feat, style_label):

        # 注意，对于数据集中的风格样本，不能使用其resnet特征，其中resnet特征实际上是随机数
        # 不需要obj_vec以及随机选择resnet特征和obj特征的部分
        feat_vec = self.resnet_feat_encoder(image_feat)
        logit = self.decode(feat_vec, style_label, cap[:, :-1], cap_len-1)
        # 为了复用train_stylelabel的代码，返回三个随机tensor
        return logit, torch.rand(feat_vec.shape).to(device), torch.rand(feat_vec.shape).to(device), torch.rand(feat_vec.shape).to(device)

    def generate_onesample(self, feat_vec, style_label):

        sentence = self.p_decoder.greedy(feat_vec, style_label)
        return sentence

    def generate_beamsearch(self, feat_vec, style_label):

        sentence, sentences = self.p_decoder.beam_search(feat_vec, style_label)
        return sentence, sentences
