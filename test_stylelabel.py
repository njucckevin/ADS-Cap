# 最终test
import torch
import os
import csv
import pickle
import json
from tqdm import tqdm

from models.Stylelabel_cont.vae_framework import EncDec_Framework
from data_load import data_load
from config import config
from utils.eval_metrics_stylelabel import generate_sen_stylelabel, generate_sen_stylelabel_reval, generate_sen_stylelabel_test, eval_pycoco, eval_ppl, eval_cls
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

step = config.step

model = EncDec_Framework(config).to(device)
log_path = config.log_dir.format(config.id)
best_model_path = log_path + '/model/model_' + str(step) + '.pt'
model.load_state_dict(torch.load(best_model_path))
model.eval()
"""
# 观察隐空间区域划分的大小
num = [0, 0, 0, 0, 0]
softmax = torch.nn.Softmax(dim=0)
for i in tqdm(range(100000)):
    latent_vec = torch.randn(1, config.latent_dim).to(device)
    pro = softmax(model_sc(latent_vec)[0])
    value, index = torch.max(pro, 0)
    if value > 0.9:
        num[int(index)] += 1
print(num)
input()
"""
generate_sen_stylelabel_reval(config, model, step, 'test')
pycoco_ro, pycoco_fu, pycoco_pos, pycoco_neg = eval_pycoco(config, step, 'test')
ppl_ro, ppl_fu, ppl_pos, ppl_neg = eval_ppl(config, step, 'test')
cls_ro, cls_fu, cls_pos, cls_neg = eval_cls(config, step, 'test')
pycoco = [pycoco_ro, pycoco_fu, pycoco_pos, pycoco_neg]
ppl = [ppl_ro, ppl_fu, ppl_pos, ppl_neg]
cls = [cls_ro, cls_fu, cls_pos, cls_neg]

results = []
pycoco_list = ["Bleu_1", "Bleu_3", "METEOR", "CIDEr"]
for i in range(4):
    result = []
    for item in pycoco_list:
        print(item+": "+str(pycoco[i][item]))
        result.append(round(pycoco[i][item]*100, 1))
    print("ppl: "+str(ppl[i]))
    result.append(round(ppl[i], 1))
    print("cls: "+str(cls[i]))
    result.append(round(cls[i]*100, 1))
    results.append(result)

path = os.path.join(log_path, 'generated/'+str(step)+'_acc.csv')
with open(path, 'w', encoding='utf-8') as f:
    csv_writer = csv.writer(f)
    metrics = ["", "Bleu_1", "Bleu_3", "METEOR", "CIDEr", "ppl", "cls"]
    styles = ["Romantic", "Humorous", "Positive", "Negative"]
    csv_writer.writerow(metrics)
    for i in range(4):
        row = [styles[i]]
        row += results[i]
        csv_writer.writerow(row)
f.close()



