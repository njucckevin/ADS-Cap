# 整理senticap的数据，统一处理为：{"captions":[{"raw":xxx, "caption":xxx, "caption_style":xxx}, ...], "style":xxx, "filename":xxx, "split":xxx}的格式
# raw_style_dict是MemCap中为四个数据集提取出的风格部分
import json
import random
from tqdm import tqdm

senticap_data_dir = '../data/senticap.json'
senticap_data = json.load(open(senticap_data_dir, 'r'))["images"]

senticap_train = [item for item in senticap_data if item["split"] == "train"]
senticap_val = [item for item in senticap_data if item["split"] == "val"]
senticap_test = [item for item in senticap_data if item["split"] == "test"]

random.shuffle(senticap_val)
senticap_train = senticap_train+senticap_val[100:]
senticap_val = senticap_val[:100]
senticap_test = senticap_test

dataset_senticap = []

raw_style_dict = json.load(open('../data/raw_style_dict.json', 'r'))
raw_not_found = []

for step, item in tqdm(enumerate(senticap_train)):
    file_name = item["filename"]
    sentences_pos = []
    sentences_neg = []
    for sentence in item["sentences"]:
        if sentence["sentiment"] == 1:
            raw = sentence["raw"]
            if raw in raw_style_dict:
                caption_style = raw_style_dict[raw]
                caption_style = [word.lower() for word in caption_style]
            else:
                print("not found")
                raw_not_found.append(raw)
                caption_style = []
            tokens = [word.replace("-", "") for word in sentence["tokens"]]
            if caption_style == []:
                print("empty")
            item_new = {"raw": raw, "caption": tokens, "caption_style": caption_style}
            sentences_pos.append(item_new)
        else:
            raw = sentence["raw"]
            if raw in raw_style_dict:
                caption_style = raw_style_dict[raw]
                caption_style = [word.lower() for word in caption_style]
            else:
                print("not found")
                raw_not_found.append(raw)
                caption_style = []
            tokens = [word.replace("-", "") for word in sentence["tokens"]]
            if caption_style == []:
                print("empty")
            item_new = {"raw": raw, "caption": tokens, "caption_style": caption_style}
            sentences_neg.append(item_new)
    if len(sentences_pos):
        item_new = {"captions": sentences_pos, "style": "positive", "filename": file_name, "split": "train"}
        dataset_senticap.append(item_new)
    if len(sentences_neg):
        item_new = {"captions": sentences_neg, "style": "negative", "filename": file_name, "split": "train"}
        dataset_senticap.append(item_new)

for step, item in tqdm(enumerate(senticap_val)):
    file_name = item["filename"]
    sentences_pos = []
    sentences_neg = []
    for sentence in item["sentences"]:
        if sentence["sentiment"] == 1:
            raw = sentence["raw"]
            if raw in raw_style_dict:
                caption_style = raw_style_dict[raw]
                caption_style = [word.lower() for word in caption_style]
            else:
                print("not found")
                raw_not_found.append(raw)
                caption_style = []
            tokens = [word.replace("-", "") for word in sentence["tokens"]]
            if caption_style == []:
                print("empty")
            item_new = {"raw": raw, "caption": tokens, "caption_style": caption_style}
            sentences_pos.append(item_new)
        else:
            raw = sentence["raw"]
            if raw in raw_style_dict:
                caption_style = raw_style_dict[raw]
                caption_style = [word.lower() for word in caption_style]
            else:
                print("not found")
                raw_not_found.append(raw)
                caption_style = []
            tokens = [word.replace("-", "") for word in sentence["tokens"]]
            if caption_style == []:
                print("empty")
            item_new = {"raw": raw, "caption": tokens, "caption_style": caption_style}
            sentences_neg.append(item_new)
    if len(sentences_pos):
        item_new = {"captions": sentences_pos, "style": "positive", "filename": file_name, "split": "val"}
        dataset_senticap.append(item_new)
    if len(sentences_neg):
        item_new = {"captions": sentences_neg, "style": "negative", "filename": file_name, "split": "val"}
        dataset_senticap.append(item_new)

for step, item in tqdm(enumerate(senticap_test)):
    file_name = item["filename"]
    sentences_pos = []
    sentences_neg = []
    for sentence in item["sentences"]:
        if sentence["sentiment"] == 1:
            raw = sentence["raw"]
            if raw in raw_style_dict:
                caption_style = raw_style_dict[raw]
                caption_style = [word.lower() for word in caption_style]
            else:
                print("not found")
                raw_not_found.append(raw)
                caption_style = []
            tokens = [word.replace("-", "") for word in sentence["tokens"]]
            item_new = {"raw": raw, "caption": tokens, "caption_style": caption_style}
            sentences_pos.append(item_new)
        else:
            raw = sentence["raw"]
            if raw in raw_style_dict:
                caption_style = raw_style_dict[raw]
                caption_style = [word.lower() for word in caption_style]
            else:
                print("not found")
                raw_not_found.append(raw)
                caption_style = []
            tokens = [word.replace("-", "") for word in sentence["tokens"]]
            item_new = {"raw": raw, "caption": tokens, "caption_style": caption_style}
            sentences_neg.append(item_new)
    if len(sentences_pos):
        item_new = {"captions": sentences_pos, "style": "positive", "filename": file_name, "split": "test"}
        dataset_senticap.append(item_new)
    if len(sentences_neg):
        item_new = {"captions": sentences_neg, "style": "negative", "filename": file_name, "split": "test"}
        dataset_senticap.append(item_new)

print("Num of pos test: "+str(len([item for item in dataset_senticap if item["style"] == "positive" and item["split"] == "test"])))
print("Num of neg test: "+str(len([item for item in dataset_senticap if item["style"] == "negative" and item["split"] == "test"])))
num = 0
for item in dataset_senticap:
    num += len(item["captions"])
print("Total num of sentences: "+str(num))

print("Raw not found: "+str(len(raw_not_found)))

out_dir = '../data/dataset_senticap.json'
with open(out_dir, 'w') as f:
    json.dump(dataset_senticap, f)


