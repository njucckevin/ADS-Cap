# 将flickrstyle和senticap的ref的格式转化为便于pycoco直接计算指标的格式，用于val和test时计算Bleu等指标
import json
from tqdm import tqdm
"""
eval_text_dir = '../data/dataset_ro_test.json'
cocoeval_out_dir = '../data/pycocoref_ro_test.json'

eval_dataset = json.load(open(eval_text_dir, 'r'))

cocoeval_out = {}
for step, item in tqdm(enumerate(eval_dataset)):
    refs = []
    ref = {}
    ref['image_id'] = step
    ref['id'] = step
    caption = ''
    sentence = item['caption']
    for word in sentence:
        caption += word
        caption += ' '
    ref['caption'] = caption[:-1]
    refs.append(ref)
    cocoeval_out[step] = refs

print(len(cocoeval_out))
with open(cocoeval_out_dir, 'w') as f:
    json.dump(cocoeval_out, f)

"""
eval_text_dir = '../data/dataset_pos_test.json'
cocoeval_out_dir = '../data/pycocoref_pos_test.json'

eval_dataset = json.load(open(eval_text_dir, 'r'))

cocoeval_out = {}
for step, item in tqdm(enumerate(eval_dataset)):
    refs = []
    for step_new, item_new in enumerate(item["caption"]):
        ref = {}
        ref['image_id'] = step
        ref['id'] = step_new
        caption = ''
        sentence = item_new
        for word in sentence:
            caption += word
            caption += ' '
        ref['caption'] = caption[:-1]
        refs.append(ref)
    cocoeval_out[step] = refs

print(len(cocoeval_out))
with open(cocoeval_out_dir, 'w') as f:
    json.dump(cocoeval_out, f)

"""
# 用于测试普通NIC模型在senticap上的指标
dataset_senticap_dir = '../data/dataset_senticap.json'
dataset_senticap = json.load(open(dataset_senticap_dir, 'r'))
cocoeval_out_pos = {}
cocoeval_out_neg = {}
i_pos = 0
i_neg = 0
for step, item in tqdm(enumerate(dataset_senticap)):
    if item["split"] == "test":
        if item["style"] == "positive":
            refs = []
            ref = {}
            ref['image_id'] = i_pos
            ref['caption'] = item["nic"]
            ref['id'] = i_pos
            refs.append(ref)
            cocoeval_out_pos[i_pos] = refs
            i_pos += 1
        elif item["style"] == "negative":
            refs = []
            ref = {}
            ref['image_id'] = i_neg
            ref['caption'] = item["nic"]
            ref['id'] = i_neg
            refs.append(ref)
            cocoeval_out_neg[i_neg] = refs
            i_neg += 1
print("Num of pos: "+str(i_pos))
print("num of neg: "+str(i_neg))
with open('../data/pycocoref_nic_pos.json', 'w') as f:
    json.dump(cocoeval_out_pos, f)
with open('../data/pycocoref_nic_neg.json', 'w') as f:
    json.dump(cocoeval_out_neg, f)
"""