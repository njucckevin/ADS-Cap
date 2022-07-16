# 利用风格分类任务引导vae训练过程，实现根据风格划分隐空间
# 隐变量由两部分组成：描述对象和描述方式
import torch
import random
import numpy as np

import time
import sys
import os
import shutil
import json
import pickle
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from config import config
from data_load import data_load, data_load_ws
from models.CVAE_cont.vae_framework import VAE_Framework
from models.Hidden_analyzer.style_classifier import Style_Classifier
from utils.tensorboard_writer import write_scalar, write_metrics
from utils.loss import CE_KL_weighted_1, Style_loss_weighted, CE_selected, KL_selected_1, Style_loss_selected
from utils.loss_cont import Contrastive_loss
from utils.log_print import train_print
from utils.eval_metrics_cvae import generate_sen, eval_pycoco, eval_ppl, eval_cls
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

seed = config.seed
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

log_path = config.log_dir.format(config.id)
if not os.path.exists(log_path):
    os.makedirs(log_path)
para_path = os.path.join(log_path, 'para.json')
with open(para_path, 'w') as f:
    json.dump(sys.argv, f)
shutil.copy('./config.py', log_path)

epochs = config.epoch
global_step = 0
writer = SummaryWriter(log_path)

with open(config.vocab, 'rb') as f:
    vocab = pickle.load(f)

train_loader = data_load(config, 'train', config.train)

model = VAE_Framework(config).to(device)
style_classifier = Style_Classifier(config).to(device)

weight_loss_dict = {0: 5, 1: 10, 2: 15, 3: 20, 4: 1}

if config.finetune:
    pretrain_model_path = config.log_dir.format(config.pretrain_id)+'/model/model_'+str(config.pretrain_step)+'.pt'
    pretrain_model_path_sc = config.log_dir.format(config.pretrain_id)+'/model/model_sc_'+str(config.pretrain_step)+'.pt'
    model.load_state_dict(torch.load(pretrain_model_path))
    style_classifier.load_state_dict(torch.load(pretrain_model_path_sc))
    weight_dict = {0: 70, 1: 70, 2: 160, 3: 160, 4: 2}
    train_loader = data_load_ws(config, 'train', config.train, weight_dict, 15000)
    weight_loss_dict = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1}
    global_step = config.pretrain_step

optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': config.lr},
                              {'params': style_classifier.parameters(), 'lr': config.style_lr}], betas=(0.9, 0.98), eps=1e-9)

criterion_w = CE_KL_weighted_1()
criterion_style_w = Style_loss_weighted()
criterion_cont = Contrastive_loss(config)
ce_selected = CE_selected()
kl_selected = KL_selected_1()
style_loss_selected = Style_loss_selected()

loss_ce_average = 0
loss_kl_average = 0
loss_st_average = 0
loss_cos_average = 0
loss_cont_average = 0
loss_ce_style_average = 0
loss_kl_style_average = 0
loss_st_style_average = 0

kl_rate_init = config.kl_rate
st_rate_init = config.style_rate
cont_rate_init = config.cont_rate
for epoch in range(epochs):

    model.train()
    style_classifier.train()
    total_step = len(train_loader)
    epoch_time = time.time()
    step_time = time.time()
    for step, (cap, cap_len, obj, obj_num, cap_style, cap_style_len, style_label, feat) in enumerate(train_loader):

        global_step += 1
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

        logit, mu, sigma2, mu_pri, sigma2_pri, latent_vec, obj_vec, res_vec, feat_vec = model(cap, cap_len, obj, obj_num, cap_style, cap_style_len, feat, style_label)

        weight = [weight_loss_dict[int(item)] for item in style_label]  # weighted loss
        # weight = [1 for item in style_label]
        weight = torch.Tensor(weight).to(device)

        loss_ce, loss_kl = criterion_w(logit, mu, sigma2, mu_pri, sigma2_pri, cap, cap_len, weight)

        loss_ce_style = ce_selected(logit, cap, cap_len, style_label)
        loss_kl_style = kl_selected(mu, sigma2, mu_pri, sigma2_pri, style_label)

        style_pred = style_classifier(latent_vec)
        loss_st = criterion_style_w(style_pred, style_label, weight)

        loss_st_style = style_loss_selected(style_pred, style_label)

        loss_cont, loss_cos = criterion_cont(obj_vec, res_vec, style_label, feat_vec)

        cont_rate = cont_rate_init*(1+global_step/50000)  # cont_rate递增
        st_rate = st_rate_init*(0+global_step/100000)
        # st_rate = st_rate_init
        kl_rate = kl_rate_init*(0+global_step/200000) if global_step <= 200000 else kl_rate_init
        # kl_rate = kl_rate_init

        loss = loss_ce + kl_rate*loss_kl + st_rate*loss_st + cont_rate*loss_cont
        loss_ce_average += loss_ce.item()
        loss_kl_average += loss_kl.item()
        loss_st_average += loss_st.item()
        loss_cos_average += loss_cos.item()
        loss_cont_average += loss_cont.item()
        loss_ce_style_average += loss_ce_style.item()
        loss_kl_style_average += loss_kl_style.item()
        loss_st_style_average += loss_st_style.item()

        model.zero_grad()
        style_classifier.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), config.grad_clip)
        nn.utils.clip_grad_value_(style_classifier.parameters(), config.grad_clip)
        optimizer.step()

        if global_step % config.save_loss_freq == 0:
            write_scalar(writer, 'loss_ce', (loss_ce_average/config.save_loss_freq), global_step)
            write_scalar(writer, 'loss_kl', (loss_kl_average/config.save_loss_freq), global_step)
            write_scalar(writer, 'loss_st', (loss_st_average/config.save_loss_freq), global_step)
            write_scalar(writer, 'loss_cos', (loss_cos_average/config.save_loss_freq), global_step)
            write_scalar(writer, 'loss_cont', (loss_cont_average/config.save_loss_freq), global_step)
            write_scalar(writer, 'loss_ce_style', (loss_ce_style_average/config.save_loss_freq), global_step)
            write_scalar(writer, 'loss_kl_style', (loss_kl_style_average/config.save_loss_freq), global_step)
            write_scalar(writer, 'loss_st_style', (loss_st_style_average/config.save_loss_freq), global_step)
            loss_ce_average = 0
            loss_kl_average = 0
            loss_st_average = 0
            loss_cos_average = 0
            loss_cont_average = 0
            loss_ce_style_average = 0
            loss_kl_style_average = 0
            loss_st_style_average = 0
        # print training information
        train_print(loss.item(), step, total_step, epoch, time.time() - step_time, time.time() - epoch_time)
        step_time = time.time()

        if global_step % config.save_model_freq == 0:
            print("Evaluating...")
            model.eval()
            style_classifier.eval()

            # 保存模型
            model_path = os.path.join(log_path, 'model')
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            save_path = os.path.join(model_path, f'model_{global_step}.pt')
            torch.save(model.state_dict(), save_path)
            save_path = os.path.join(model_path, f'model_sc_{global_step}.pt')
            torch.save(style_classifier.state_dict(), save_path)

            # val
            generate_sen(config, model, style_classifier, global_step, 'val')
            pycoco_ro, pycoco_fu, pycoco_pos, pycoco_neg = eval_pycoco(config, global_step, 'val')
            ppl_ro, ppl_fu, ppl_pos, ppl_neg = eval_ppl(config, global_step, 'val')
            cls_ro, cls_fu, cls_pos, cls_neg = eval_cls(config, global_step, 'val')
            write_metrics(writer, pycoco_ro, pycoco_fu, pycoco_pos, pycoco_neg, ppl_ro, ppl_fu, ppl_pos, ppl_neg, cls_ro, cls_fu, cls_pos, cls_neg, global_step)

            model.train()
            style_classifier.train()
