# 利用【训练集】训练风格判别器判断一个caption是否带有风格
# 利用data_load_ws，训练哪种风格时就只sample事实性+对应风格的样本
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
from utils.log_print import train_print

from config import config
from data_load import data_load_ws, data_load
from models.Discriminator.discriminator_lstm import Discriminator_lstm
from utils.tensorboard_writer import write_scalar
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

epochs = 200
global_step = 0
writer = SummaryWriter(log_path)

with open(config.vocab, 'rb') as f:
    vocab = pickle.load(f)

weight_dict = {0: 70, 1: 70, 2: 160, 3: 160, 4: 2}
train_loader = data_load_ws(config, 'train', config.train, weight_dict)
print("Num of train_loader: "+str(len(train_loader)))

model = Discriminator_lstm(config).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, betas=(0.9, 0.98), eps=1e-9)
criterion = nn.CrossEntropyLoss()

loss_d_average = 0
for epoch in range(epochs):

    total_step = len(train_loader)
    model.train()
    epoch_time = time.time()
    step_time = time.time()

    for step, (cap, cap_len, obj, obj_num, cap_style, cap_style_len, style_label, feat) in enumerate(train_loader):

        global_step += 1
        cap = cap.to(device)
        cap_len = cap_len.to(device)
        style_label = style_label.to(device)
        style_label = torch.Tensor([1 if int(item) != 4 else 0 for item in style_label]).to(device)
        cap_len = cap_len + 2  # 开始符结束符

        style_pred = model(cap, cap_len)

        loss_d = criterion(style_pred, style_label.long())
        loss_d_average += loss_d.item()

        model.zero_grad()
        loss_d.backward()
        nn.utils.clip_grad_value_(model.parameters(), config.grad_clip)
        optimizer.step()

        train_print(loss_d.item(), step, total_step, epoch, time.time() - step_time, time.time() - epoch_time)
        step_time = time.time()

        if global_step % config.save_loss_freq == 0:
            write_scalar(writer, 'loss_d', (loss_d_average / config.save_loss_freq), global_step)
            loss_d_average = 0

        if global_step % config.save_model_freq == 0:
            print("Evaluating...")

            model.eval()
            model_path = os.path.join(log_path, 'model')
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            save_path = os.path.join(model_path, f'model_{global_step}.pt')
            torch.save(model.state_dict(), save_path)

            val_dir = [config.val_ro, config.val_fu, config.val_pos, config.val_neg]
            for i in range(len(val_dir)):
                eval_loader = data_load(config, 'train', val_dir[i])
                print("Num of eval_loader: " + str(len(eval_loader)))
                total_eval = len(eval_loader)
                correct_num = 0

                for step, (cap, cap_len, obj, obj_num, cap_style, cap_style_len, style_label, feat) in enumerate(eval_loader):
                    cap = cap.to(device)
                    cap_len = cap_len.to(device)
                    style_label = style_label.to(device)
                    cap_len = cap_len + 2  # 开始符结束符
                    style_pred = model(cap, cap_len)
                    pred_id = style_pred[0].argmax()
                    if int(pred_id) == 1:
                        correct_num += 1

                print("val_acc_"+str(i)+": "+str(correct_num/total_eval))
                write_scalar(writer, 'val_acc_'+str(i), (correct_num/total_eval), global_step)

            model.train()
