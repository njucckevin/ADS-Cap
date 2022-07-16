# 为预处理好的json数据集生成单词表
import argparse
import json
import pickle
import sys
from tqdm import tqdm

sys.path.append('.')
from utils.vocab import Vocabulary

vocab = Vocabulary()

dataset_coco = json.load(open('../data/dataset_coco.json', 'r'))['images']
dataset_flickrstyle = json.load(open('../data/dataset_flickrstyle.json', 'r'))
dataset_senticap = json.load(open('../data/dataset_senticap.json', 'r'))

counter_coco = {}
for item in tqdm(dataset_coco):
    if item["split"] != "train":
        continue
    for sentence in item["sentences"]:
        sentence_token = sentence["tokens"]
        for token in sentence_token:
            counter_coco[token] = counter_coco.get(token, 0)+1
cand_word_coco = [token for token, f in counter_coco.items() if f > 5]
print("Vocab size coco: "+str(len(cand_word_coco)))

counter_flickrstyle = {}
for item in tqdm(dataset_flickrstyle):
    if item["split"] != "train":
        continue
    sentence_token = item["caption"]
    for token in sentence_token:
        counter_flickrstyle[token] = counter_flickrstyle.get(token, 0)+1
cand_word_flickrstyle = [token for token, f in counter_flickrstyle.items() if f > 0]
print("Vocab size flickrstyle: "+str(len(cand_word_flickrstyle)))

counter_senticap = {}
for item in tqdm(dataset_senticap):
    if item["split"] != "train":
        continue
    for sentence in item["captions"]:
        for token in sentence["caption"]:
            counter_senticap[token] = counter_senticap.get(token, 0)+1
cand_word_senticap = [token for token, f in counter_senticap.items() if f > 0]
print("Vocab size senticap: "+str(len(cand_word_senticap)))

cand_word = list(set(cand_word_coco+cand_word_flickrstyle+cand_word_senticap))
print("Vocab size: "+str(len(cand_word)))

for w in cand_word:
    vocab.add_word(w)
vocab_path = '../data/vocab.pkl'
with open(vocab_path, 'wb') as f:
    pickle.dump(vocab, f)
print(f'vocab size: {vocab.get_size()}, saved to {vocab_path}')
