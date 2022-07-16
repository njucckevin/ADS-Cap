# 使用paired数据：为用到的COCO和flickrstyle中图片生成resnet特征
import os
import sys
import json
import pickle
from tqdm import tqdm
import numpy as np

import torchvision.transforms as transforms
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Img_data(Dataset):

    def __init__(self, img_dir, img_files, transforms):
        super(Img_data, self).__init__()
        self.img_dir = img_dir
        self.img_files = img_files
        self.transforms = transforms

    def __getitem__(self, item):
        img_path = self.img_dir+str(self.img_files[item])
        if os.path.exists(img_path):
            image = Image.open(img_path).convert('RGB')
        else:
            print("Image not found!")
            sys.exit(0)
        image = self.transforms(image)
        return image, self.img_files[item][:-4]

    def __len__(self):
        return len(self.img_files)


def data_load(img_dir, img_files, transforms):
    dataset = Img_data(img_dir, img_files, transforms)
    data_loader = DataLoader(dataset=dataset,
                             batch_size=20,
                             shuffle=False,
                             num_workers=8,
                             )
    return data_loader


def my_transforms():
    # Transform and unify the format of images (both coco and flickrstyle)
    train_transform = transforms.Compose([
        # 重置图像分辨率，使用ANTTALIAS抗锯齿方法
        transforms.Resize([224, 224], Image.ANTIALIAS),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])
    return train_transform


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        cnn = models.resnet152()
        pre = torch.load('/home/data_ti4_c/chengkz/data/resnet152-b121ed2d.pth')
        cnn.load_state_dict(pre)
        modules = list(cnn.children())[:-2]  # 输出维度[b,2048,7,7]
        self.cnn = nn.Sequential(*modules)
        self.avgpool = nn.AvgPool2d(7)  # 平均池化，用于生成global image feature

    def forward(self, images):
        features = self.cnn(images)  # [b,2048,7,7]
        batch_size = features.shape[0]
        global_features = self.avgpool(features).view(batch_size, -1)  # 全局特征[b,2048]
        return global_features  # 最终生成的深度为2048的特征


def generate_feat(transforms, imgs_dir, data, out_dir):

    print("Num of image: "+str(len(data)))
    data_loader = data_load(imgs_dir, data, transforms)

    resnet_encoder = Encoder()
    resnet_encoder.to(device)
    resnet_encoder.eval()

    for images, file_names in tqdm(data_loader):
        images = images.to(device)
        resnet_feat = resnet_encoder(images).cpu().detach().numpy()
        for step, file_name in enumerate(file_names):
            np.save(out_dir+str(file_name), resnet_feat[step])

transforms = my_transforms()

# 准备flickrstyle的7000张图片
flickr8k_dir = '/home/data_ti4_c/chengkz/data/Flicker8k_dataset/'
flickrstyle_out_dir = '/home/data_ti4_c/chengkz/data/resnet_feat/flickrstyle/'
with open('../data/train_ro.p', 'rb') as f:
    imgs_flickrstyle = pickle.load(f)

# 准备coco train的80000+图片
coco_train_dir = '/home/data_ti4_c/chengkz/data/coco_dataset/train2014/'
coco_out_dir = '/home/data_ti4_c/chengkz/data/resnet_feat/coco/'
dataset_dir_coco = '/home/data_ti4_c/chengkz/data/caption_datasets/dataset_coco.json'
dataset_coco = json.load(open(dataset_dir_coco, 'r'))['images']
imgs_coco = []
for item in dataset_coco:
    if item['split'] != 'train':
        continue
    imgs_coco.append(item["filename"])

# 准备senticap的2225张图片
senticap_img_dir = '/home/data_ti4_c/chengkz/data/SentiCap2360/'
senticap_out_dir = '/home/chengkz/data/resnet_feat/senticap/'
if not os.path.exists(senticap_out_dir):
    os.makedirs(senticap_out_dir)
dataset_dir_senticap = '../data/dataset_senticap.json'
dataset_senticap = json.load(open(dataset_dir_senticap, 'r'))
imgs_senticap = []
for item in dataset_senticap:
    imgs_senticap.append(item["filename"])

# generate_feat(transforms, flickr8k_dir, imgs_flickrstyle, flickrstyle_out_dir)
# generate_feat(transforms, coco_train_dir, imgs_coco, coco_out_dir)
generate_feat(transforms, senticap_img_dir, imgs_senticap, senticap_out_dir)