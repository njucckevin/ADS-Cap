# 特征空间可视化

import torch
import os
import pickle
import csv
import json
from tqdm import tqdm
import random

from models.CVAE_cont.vae_framework import VAE_Framework
from models.Hidden_analyzer.style_classifier import Style_Classifier
from data_load import data_load, data_load
from config import config
if config.vis_mode == 'vis':
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from sklearn.manifold import TSNE
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

seed = config.seed
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

with open(config.vocab, 'rb') as f:
    vocab = pickle.load(f)

if config.vis_mode == 'save':
    step = config.step

    model = VAE_Framework(config).to(device)
    model_sc = Style_Classifier(config).to(device)
    log_path = config.log_dir.format(config.id)
    best_model_path = log_path + '/model/model_' + str(step) + '.pt'
    best_model_sc_path = log_path + '/model/model_sc_' + str(step) + '.pt'
    model.load_state_dict(torch.load(best_model_path))
    model_sc.load_state_dict(torch.load(best_model_sc_path))
    model.eval()
    model_sc.eval()

    print("loading data...")
    train_loader = data_load(config, 'train', config.train)
    model.train()

    num_step = 20
    feature_obj = torch.zeros(num_step*config.batch_size, config.align_dim)
    feature_res = torch.zeros(num_step*config.batch_size, config.align_dim)
    label = torch.zeros(num_step*config.batch_size)     # 标记一些特殊的场景，狗标为1，猫标为2

    with torch.no_grad():
        for step, (cap, cap_len, obj, obj_num, cap_style, cap_style_len, style_label, feat) in enumerate(tqdm(train_loader)):
            # if int(style_label[0]) == 2:
            if step >= num_step:
                break
            cap = cap.to(device)
            cap_len = cap_len.to(device)
            obj = obj.to(device)
            obj_num = obj_num.to(device)
            cap_style = cap_style.to(device)
            cap_style_len = cap_style_len.to(device)
            style_label = style_label.to(device)
            feat = feat.to(device)

            cap_len = cap_len + 2  # 开始符结束符
            obj_num = obj_num + 2
            cap_style_len = cap_style_len + 2

            logit, mu, sigma2, mu_pri, sigma2_pri, latent_vec, obj_vec, res_vec, feat_vec = model(cap, cap_len, obj, obj_num, cap_style,
                                                                              cap_style_len, feat, style_label)
            feature_obj[step*config.batch_size:step*config.batch_size+config.batch_size] = obj_vec
            feature_res[step*config.batch_size:step*config.batch_size+config.batch_size] = res_vec
            for i in range(cap.size(0)):
                if vocab.word_to_id("dog") in cap[i].tolist():
                    label[step*config.batch_size+i] = 1
                if vocab.word_to_id("plane") in cap[i].tolist():
                    label[step*config.batch_size+i] = 2

    log_path = config.log_dir.format(config.id)
    torch.save(feature_obj, log_path+'/feature_obj.pt', _use_new_zipfile_serialization=False)
    torch.save(feature_res, log_path+'/feature_res.pt', _use_new_zipfile_serialization=False)
    torch.save(label, log_path+'/label.pt', _use_new_zipfile_serialization=False)
    print("save success")

elif config.vis_mode == 'vis':

    feature_obj = torch.load('./vis/feature_obj_c0.1.pt')
    feature_res = torch.load('./vis/feature_res_c0.1.pt')
    label = torch.load('./vis/label_c0.1.pt')

    num_samples = feature_obj.size(0)

    data = torch.cat([feature_obj, feature_res], dim=0)
    print("num of samples: "+str(num_samples))
    print(data.shape)
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    result = tsne.fit_transform(data.detach().numpy())

    path = './vis/result_c0.1.csv'
    with open(path, 'w', encoding='utf-8') as f:
        csv_writer = csv.writer(f)
        for i in range(2000):
            row = list(result[i])+list(result[i+2000])+list([int(label[i])])
            csv_writer.writerow(row)
    f.close()

    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    for i in tqdm(range(num_samples)):
        s1 = plt.scatter(result[i][0], result[i][1], s=20, color='g', marker='+')
        s2 = plt.scatter(result[i+2000][0], result[i+2000][1], s=20, color='r', marker='+')

    plt.legend([s1, s2], ['obj', 'res'], loc='upper left')  # lower right
    plt.axis('off')
    plt.title("feature space")
    plt.show()

