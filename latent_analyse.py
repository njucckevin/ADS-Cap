# 隐空间可视化

import torch
import os
import pickle
import csv
import json
from tqdm import tqdm
import random

from models.CVAE_cont.vae_framework import VAE_Framework
from models.Hidden_analyzer.style_classifier import Style_Classifier
from data_load import data_load
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

"""
train_loader = data_load(config, 'train', './data/dataset_train.json')
# 观察整个训练集中的object类别
with torch.no_grad():
    counter = {}
    for step, (cap, cap_len, obj, obj_num, cap_style, cap_style_len, style_label, feat) in enumerate(
            tqdm(train_loader)):
        if int(style_label[0]) != 4:
            continue
        for noun in obj[0]:
            counter[int(noun)] = counter.get(int(noun), 0) + 1
freq_words = sorted(counter, key=counter.__getitem__, reverse=True)
for k in freq_words:
    print(str(vocab.idList_to_sent([1, k, 2])) + ": " + str(counter[k]))
object_categories = {0: ["man", "men"], 1: ["woman", "women"], 2: ["people", "crowd"], 3: ["boy", "boys"],
                     4: ["girl", "girls"], 5: ["child", "children"], 6: ["dog", "dogs"], 7: ["cat", "cats"],
                     8: ["water"], 9: ["street"], 10: ["beach"], 11: ["field"], 12: ["food"]}
input()
"""
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
    """
    # 观察重要的维度
    print(model_sc.state_dict())
    weights = model_sc.state_dict()['classifier.weight']
    k = 10
    for i in range(5):
        weight = list(weights[i])
        weight = [abs(float(item)) for item in weight]
        index_k = []
        for j in range(k):
            index_j = weight.index(max(weight))
            index_k.append(index_j)
            weight[index_j] = float('-inf')
        print(index_k)
        input()
    """
    # 生成不同风格对应的隐变量（保存所有维度）
    important_dims = torch.tensor(list(range(100))).to(device)

    print("loading data...")
    train_loader = data_load(config, 'train', './data/dataset_train_finetune.json')
    model.train()

    num_step = 50
    data = torch.zeros(num_step*config.batch_size, 100)
    label = torch.zeros(num_step*config.batch_size)
    label_sp = torch.zeros(num_step*config.batch_size)

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

            for i in range(cap.size(0)):
                if vocab.word_to_id("to") in cap_style[i].tolist() \
                        and vocab.word_to_id("meet") in cap_style[i].tolist() \
                        and vocab.word_to_id("his") in cap_style[i].tolist() \
                        and vocab.word_to_id("lover") in cap_style[i].tolist():
                    label_sp[step*config.batch_size+i] = 1
                if vocab.word_to_id("with") in cap_style[i].tolist() \
                        and vocab.word_to_id("full") in cap_style[i].tolist() \
                        and vocab.word_to_id("joy") in cap_style[i].tolist():
                    label_sp[step*config.batch_size+i] = 2
                if vocab.word_to_id("in") in cap_style[i].tolist()\
                        and vocab.word_to_id("love") in cap_style[i].tolist():
                    print(cap_style[i].tolist())
                    input()
                    label_sp[step*config.batch_size+i] = 3
                if vocab.word_to_id("determined") in cap_style[i].tolist() \
                        and vocab.word_to_id("to") in cap_style[i].tolist() \
                        and vocab.word_to_id("win") in cap_style[i].tolist():
                    label_sp[step*config.batch_size+i] = 4
            data[step*config.batch_size:step*config.batch_size+config.batch_size] = torch.index_select(latent_vec, 1, important_dims)
            label[step*config.batch_size:step*config.batch_size+config.batch_size] = style_label

    log_path = config.log_dir.format(config.id)
    torch.save(data, log_path+'/latent_vec.pt', _use_new_zipfile_serialization=False)
    torch.save(label, log_path+'/label_latent.pt', _use_new_zipfile_serialization=False)
    torch.save(label_sp, log_path+'/label_latent_sp.pt', _use_new_zipfile_serialization=False)
    print("save success")

elif config.vis_mode == 'vis':
    # important_dims = torch.tensor([34, 68, 84, 77, 7, 85, 31, 70, 16, 76, 15])
    important_dims = torch.tensor(list(range(100)))
    data = torch.load('./vis/latent_vec.pt')
    label = torch.load('./vis/label_latent.pt')
    label_sp = torch.load('./vis/label_latent_sp.pt')

    num_samples = data.size(0)

    data = torch.index_select(data, 1, important_dims)
    print("num of samples: "+str(num_samples))
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    result = tsne.fit_transform(data.detach().numpy())

    styles = ["ro", "fu", "pos", "neg"]
    for i, style in enumerate(styles):
        path = './vis/'+style+'_result.csv'
        with open(path, 'w', encoding='utf-8') as f:
            csv_writer = csv.writer(f)
            for j, item in enumerate(result):
                if label[j] == i:
                    csv_writer.writerow(list(item)+list([int(label_sp[j])]))
        f.close()

    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    for i in tqdm(range(num_samples)):
        if label[i] == 0:
            s1 = plt.scatter(result[i][0], result[i][1], s=20, color='g', marker='+')
        elif label[i] == 1:
            s2 = plt.scatter(result[i][0], result[i][1], s=20, color='r', marker='+')
        elif label[i] == 2:
            s3 = plt.scatter(result[i][0], result[i][1], s=20, color='b', marker='+')
        elif label[i] == 3:
            s4 = plt.scatter(result[i][0], result[i][1], s=20, color='brown', marker='+')
        #elif label[i] == 4:
        #    s5 = plt.scatter(result[i][0], result[i][1], s=15, color='pink')

    plt.legend([s1, s2, s3, s4], ['romantic', 'humorous', 'positive', 'negative', 'factual'], loc='upper left')  # lower right
    plt.axis('off')
    plt.title("latent space")
    plt.show()

