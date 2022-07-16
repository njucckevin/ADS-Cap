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

        # 注意，数据集给出了每个样本的object word和resnet特征，但是其中对于风格样本，不能使用其resnet特征；对于coco样本，随机使用object word特征和resnet特征
        obj_vec = self.objword_feat_encoder(obj)
        res_vec = self.resnet_feat_encoder(image_feat)
        samples_style = torch.Tensor([1 if int(item) != 4 else 0 for item in style_label]).to(device)
        random_weight_obj = torch.Tensor([1 if samples_style[i] == 1 else self.weight_feat_dict[random.randint(0, 1)] for i in range(cap.size(0))]).to(device)
        random_weight_res = 1-random_weight_obj
        feat_vec = obj_vec*(random_weight_obj.unsqueeze(1).expand(obj_vec.shape))+res_vec*(random_weight_res.unsqueeze(1).expand(res_vec.shape))

        logit = self.decode(feat_vec, style_label, cap[:, :-1], cap_len-1)
        return logit, obj_vec, res_vec, feat_vec

    def generate_onesample(self, feat_vec, style_label):

        sentence = self.p_decoder.greedy(feat_vec, style_label)
        return sentence

    def generate_beamsearch(self, feat_vec, style_label):

        sentence, sentences = self.p_decoder.beam_search(feat_vec, style_label)
        return sentence, sentences
