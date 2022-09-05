# stylelabel（不使用CVAE）的baseline
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
from models.Stylelabel_cont.vae_framework import EncDec_Framework
from utils.tensorboard_writer import write_scalar, write_metrics
from utils.loss import CE_weighted
from utils.loss_cont import Contrastive_loss
from utils.log_print import train_print
from utils.eval_metrics_stylelabel import generate_sen_stylelabel, eval_pycoco, eval_ppl, eval_cls
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

model = EncDec_Framework(config).to(device)

weight_loss_dict = {0: 5, 1: 10, 2: 15, 3: 20, 4: 1}

if config.finetune:
    pretrain_model_path = config.log_dir.format(config.pretrain_id)+'/model/model_'+str(config.pretrain_step)+'.pt'
    model.load_state_dict(torch.load(pretrain_model_path))
    weight_dict = {0: 70, 1: 70, 2: 160, 3: 160, 4: 2}
    train_loader = data_load_ws(config, 'train', config.train, weight_dict, 15000)
    weight_loss_dict = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1}
    global_step = config.pretrain_step

optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': config.lr}], betas=(0.9, 0.98), eps=1e-9)

criterion_w = CE_weighted()
criterion_cont = Contrastive_loss(config)

loss_ce_average = 0
loss_cos_average = 0
loss_cont_average = 0

cont_rate_init = config.cont_rate
for epoch in range(epochs):

    model.train()
    total_step = len(train_loader)
    epoch_time = time.time()
    step_time = time.time()
    for step, (cap, cap_len, obj, obj_num, cap_style, cap_style_len, style_label, feat) in enumerate(train_loader):

        global_step += 1
        cap = cap.to(device)
        cap_len = cap_len.to(device)
        obj = obj.to(device)
        obj_num = obj_num.to(device)
        style_label = style_label.to(device)
        feat = feat.to(device)

        cap_len = cap_len + 2  # 开始符结束符
        obj_num = obj_num + 2

        logit, obj_vec, res_vec, feat_vec = model(cap, cap_len, obj, obj_num, feat, style_label)
        weight = [weight_loss_dict[int(item)] for item in style_label]  # weighted loss
        # weight = [1 for item in style_label]
        weight = torch.Tensor(weight).to(device)
        loss_ce = criterion_w(logit, cap, cap_len, weight)

        loss_cont, loss_cos = criterion_cont(obj_vec, res_vec, style_label, feat_vec)

        cont_rate = cont_rate_init*(1+global_step/50000)  # cont_rate递增

        loss = loss_ce + cont_rate*loss_cont
        loss_ce_average += loss_ce.item()
        loss_cos_average += loss_cos.item()
        loss_cont_average += loss_cont.item()

        model.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), config.grad_clip)
        optimizer.step()

        if global_step % config.save_loss_freq == 0:
            write_scalar(writer, 'loss_ce', (loss_ce_average/config.save_loss_freq), global_step)
            write_scalar(writer, 'loss_cos', (loss_cos_average/config.save_loss_freq), global_step)
            write_scalar(writer, 'loss_cont', (loss_cont_average/config.save_loss_freq), global_step)
            loss_ce_average = 0
            loss_cos_average = 0
            loss_cont_average = 0
        # print training information
        train_print(loss.item(), step, total_step, epoch, time.time() - step_time, time.time() - epoch_time)
        step_time = time.time()

        if global_step % config.save_model_freq == 0:
            print("Evaluating...")
            model.eval()

            # 保存模型
            model_path = os.path.join(log_path, 'model')
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            save_path = os.path.join(model_path, f'model_{global_step}.pt')
            torch.save(model.state_dict(), save_path)

            # val
            generate_sen_stylelabel(config, model, global_step, 'val')
            pycoco_ro, pycoco_fu, pycoco_pos, pycoco_neg = eval_pycoco(config, global_step, 'val')
            ppl_ro, ppl_fu, ppl_pos, ppl_neg = eval_ppl(config, global_step, 'val')
            cls_ro, cls_fu, cls_pos, cls_neg = eval_cls(config, global_step, 'val')
            write_metrics(writer, pycoco_ro, pycoco_fu, pycoco_pos, pycoco_neg, ppl_ro, ppl_fu, ppl_pos, ppl_neg, cls_ro, cls_fu, cls_pos, cls_neg, global_step)

            model.train()
