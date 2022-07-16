# 整合flickrstyle+senticap+coco，形成train、val和test的数据集
# 每个item：{"caption":xxx, "style":xxx, "objects":xxx, "caption_style":xxx, "filename":xxx, "split":xxx}
# 这里的val和test数据集的"caption"中将包含多个ref
# 而对于train，"caption"只有一个
import json
import pickle
from tqdm import tqdm
import random
import nltk

dataset_flickrstyle_dir = '../data/dataset_flickrstyle.json'
dataset_senticap_dir = '../data/dataset_senticap.json'
dataset_flickrstyle = json.load(open(dataset_flickrstyle_dir, 'r'))
dataset_senticap = json.load(open(dataset_senticap_dir, 'r'))
dataset_dir_coco = '../data/dataset_coco.json'
dataset_coco = json.load(open(dataset_dir_coco, 'r'))['images']

objects_dir = '../data/objects_vocab_plus.json'
objects_vocab = json.load(open(objects_dir, 'r'))

dataset_train = []
dataset_ro_val = []
dataset_fu_val = []
dataset_ro_test = []
dataset_fu_test = []
dataset_pos_val = []
dataset_neg_val = []
dataset_pos_test = []
dataset_neg_test = []

print("Processing flickrstyle...")
for item in tqdm(dataset_flickrstyle):
    filename = item["filename"]
    if item["split"] == "train":  # train时object word来自本身caption
        sentence = item["caption"]
        if '' in sentence:
            sentence.remove('')
        words_pos = nltk.pos_tag(sentence, lang='eng')
        noun_words = [word for word, value in words_pos if value in ['NN', 'NNP', 'NNS'] and word in objects_vocab]
        item_new = {"caption": item["caption"], "style": item["style"],
                    "objects": noun_words, "caption_style": item["caption_style"], "filename": filename, "split": item["split"]}
        dataset_train.append(item_new)
    else:  # val和test时object word来自生成的nic
        item_new = {"caption": item["caption"], "style": item["style"],
                    "objects": [], "caption_style": [], "filename": filename, "split": item["split"]}
        if item_new["style"] == "romantic" and item_new["split"] == "val":
            dataset_ro_val.append(item_new)
        elif item_new["style"] == "romantic" and item_new["split"] == "test":
            dataset_ro_test.append(item_new)
        elif item_new["style"] == "funny" and item_new["split"] == "val":
            dataset_fu_val.append(item_new)
        elif item_new["style"] == "funny" and item_new["split"] == "test":
            dataset_fu_test.append(item_new)

print("Processing coco...")
for item in tqdm(dataset_coco):
    if item['split'] != 'train':
        continue
    filename = item["filename"]
    for sentence in item['sentences']:
        sentence_token = sentence['tokens']
        if '' in sentence_token:
            sentence_token.remove('')
        words_pos = nltk.pos_tag(sentence_token, lang='eng')
        noun_words = [word for word, value in words_pos if value in ['NN', 'NNP', 'NNS']]
        item_new = {"caption": sentence_token, "style": "factual",
                    "objects": noun_words, "caption_style": [], "filename": filename, "split": "train"}
        dataset_train.append(item_new)

print("Processing senticap...")
for item in tqdm(dataset_senticap):
    filename = item["filename"]
    if item["split"] == "train":
        for sentence in item["captions"]:
            sentence_token = sentence["caption"]
            if '' in sentence_token:
                sentence_token = sentence_token.remove('')
            words_pos = nltk.pos_tag(sentence_token, lang='eng')
            noun_words = [word for word, value in words_pos if value in ['NN', 'NNP', 'NNS']]
            item_new = {"caption": sentence["caption"], "style": item["style"],
                        "objects": noun_words, "caption_style": sentence["caption_style"], "filename": filename, "split": item["split"]}
            dataset_train.append(item_new)
    else:
        captions = [sentence["caption"] for sentence in item["captions"]]
        item_new = {"caption": captions, "style": item["style"],
                    "objects": [], "caption_style": [], "filename": filename, "split": item["split"]}
        if item_new["style"] == "positive" and item_new["split"] == "val":
            dataset_pos_val.append(item_new)
        elif item_new["style"] == "positive" and item_new["split"] == "test":
            dataset_pos_test.append(item_new)
        elif item_new["style"] == "negative" and item_new["split"] == "val":
            dataset_neg_val.append(item_new)
        elif item_new["style"] == "negative" and item_new["split"] == "test":
            dataset_neg_test.append(item_new)

print("Num of train: (with empty)"+str(len(dataset_train)))
dataset_train = [item for item in dataset_train if (item["caption_style"] != [] or item["style"] == "factual")]
print("Num of train: (without empty)"+str(len(dataset_train)))

print("Num of ro val: "+str(len(dataset_ro_val)))
print("Num of ro test: "+str(len(dataset_ro_test)))
print("Num of fu val: "+str(len(dataset_fu_val)))
print("Num of fu test: "+str(len(dataset_fu_test)))
print("Num of pos val: "+str(len(dataset_pos_val)))
print("Num of pos test: "+str(len(dataset_pos_test)))
print("Num of neg val: "+str(len(dataset_neg_val)))
print("Num of neg test: "+str(len(dataset_neg_test)))

with open('../data/dataset_train.json', 'w') as f:
    json.dump(dataset_train, f)
with open('../data/dataset_ro_val.json', 'w') as f:
    json.dump(dataset_ro_val, f)
with open('../data/dataset_ro_test.json', 'w') as f:
    json.dump(dataset_ro_test, f)
with open('../data/dataset_fu_val.json', 'w') as f:
    json.dump(dataset_fu_val, f)
with open('../data/dataset_fu_test.json', 'w') as f:
    json.dump(dataset_fu_test, f)
with open('../data/dataset_pos_val.json', 'w') as f:
    json.dump(dataset_pos_val, f)
with open('../data/dataset_pos_test.json', 'w') as f:
    json.dump(dataset_pos_test, f)
with open('../data/dataset_neg_val.json', 'w') as f:
    json.dump(dataset_neg_val, f)
with open('../data/dataset_neg_test.json', 'w') as f:
    json.dump(dataset_neg_test, f)

"""
dataset_train = json.load(open('../data/dataset_train.json', 'r'))
dataset_train_nic = []
for item in dataset_train:
    if item['style'] == 'factual':
        dataset_train_nic.append(item)
print("Num of train nic: "+str(len(dataset_train_nic)))
with open('../data/dataset_train_nic.json', 'w') as f:
    json.dump(dataset_train_nic, f)
"""
"""
# 检查
dataset_train = json.load(open('../data/dataset_train.json', 'r'))
for item in tqdm(dataset_train):
    for word in item["caption_style"]:
        if word not in item["caption"]:
            print("error")
            print(word)
            print(item)
            input()
"""