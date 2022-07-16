import torch
import json
import os
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from utils.vocab import Vocabulary

class Sen_Object_data(Dataset):

    def __init__(self, config, dir, mode):
        super(Sen_Object_data, self).__init__()
        self.config = config
        self.text = json.load(open(dir, 'r'))
        self.img_dir_coco = os.path.join(config.resnet_feat_dir, 'coco')
        self.img_dir_flickrstyle = os.path.join(config.resnet_feat_dir, 'flickrstyle')
        self.img_dir_senticap = os.path.join(config.resnet_feat_dir, 'senticap')
        with open(self.config.vocab, 'rb') as f:
            self.vocab = pickle.load(f)
        self.stylelabel_dict = {"romantic": 0, "funny": 1, "positive": 2, "negative": 3, "factual": 4}
        self.mode = mode

    def __getitem__(self, item):
        if self.mode == "train":
            sen_token = self.text[item]['caption']
            label = self.stylelabel_dict[self.text[item]['style']]
            obj_token = self.text[item]['objects']
            caption_style_token = self.text[item]["caption_style"]
            sen_id, sen_len = self.vocab.tokenList_to_idList(sen_token, self.config.fixed_len)
            obj_id, obj_len = self.vocab.tokenList_to_idList(obj_token, self.config.fixed_len_o)
            style_id, style_len = self.vocab.tokenList_to_idList(caption_style_token, self.config.fixed_len_s)

            if label == 4:  # 训练时只有coco描述给图片辅助
                file_name = self.text[item]['filename'][:-4] + '.npy'
                img_path = os.path.join(self.img_dir_coco, file_name)
                feat = torch.Tensor(np.load(img_path))
            else:  # 风格描述随机生成feat
                feat = torch.randn(2048)

            return torch.Tensor(sen_id).long(), sen_len, torch.Tensor(obj_id).long(), obj_len, torch.Tensor(style_id).long(), style_len, label, feat
        else:
            obj_token = self.text[item]['objects']
            obj_id, obj_len = self.vocab.tokenList_to_idList(obj_token, self.config.fixed_len_o)

            label = self.stylelabel_dict[self.text[item]['style']]
            file_name = self.text[item]['filename'][:-4]+'.npy'
            if label == 0 or label == 1:
                img_path = os.path.join(self.img_dir_flickrstyle, file_name)
            elif label == 2 or label == 3:
                img_path = os.path.join(self.img_dir_senticap, file_name)
            else:
                img_path = os.path.join(self.img_dir_coco, file_name)
            feat = torch.Tensor(np.load(img_path))  # 验证和测试时风格描述也给图片辅助
            return torch.Tensor(obj_id).long(), obj_len, feat

    def __len__(self):
        return len(self.text)

def data_load(config, mode, dir):
    dataset = Sen_Object_data(config, dir, mode)
    data_loader = DataLoader(dataset=dataset,
                             batch_size=config.batch_size if mode == 'train' else 1,
                             shuffle=True if mode == 'train' else False,
                             num_workers=config.num_workers,
                             )
    return data_loader

def data_load_ws(config, mode, dir, weight_dict, num_samples):
    """使用weightedsampler对数据集进行加权采样以缓解样本数量不平衡的问题"""
    # weight_dict = {0: 40, 1: 40, 2: 80, 3: 80, 4: 1}
    print("preparing data..")
    dataset = Sen_Object_data(config, dir, mode)
    label = [weight_dict[int(style_label)] for (cap, cap_len, obj, obj_num, cap_style, cap_style_len, style_label, feat) in dataset]
    sampler = WeightedRandomSampler(label, num_samples=num_samples, replacement=False)
    data_loader = DataLoader(dataset=dataset,
                             batch_size=config.batch_size if mode == 'train' else 1,
                             sampler=sampler,
                             num_workers=config.num_workers,
                             )
    return data_loader