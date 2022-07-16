# 最终test
import torch
import os
import csv
import pickle
import json
from tqdm import tqdm

from models.CVAE_cont.vae_framework import VAE_Framework
from models.Hidden_analyzer.style_classifier import Style_Classifier
from data_load import data_load
from config import config
from utils.eval_metrics_cvae import generate_sen, generate_sen_reval, generate_sen_test, eval_pycoco, eval_ppl, eval_cls
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

generate_sen_test(config, model, model_sc, step, 'test')

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




