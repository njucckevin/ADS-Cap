# 整理flickrstyle的数据， 统一处理为：{"raw":xxx, "caption":xxx, "style":xxx, "caption_style":xxx, "filename":xxx, "split":xxx}的格式
# raw_style_dict是MemCap中为四个数据集提取出的风格部分
import json
import pickle
import random
from tqdm import tqdm
import nltk

flickrstyle_filename_dir = '../data/train_ro.p'
with open(flickrstyle_filename_dir, 'rb') as f:
    flickrstyle_filename_data = pickle.load(f)  # 7000张图片的filename

dataset_dir_ro = '../data/romantic_train.txt'
dataset_dir_fu = '../data/funny_train.txt'
dataset_ro = []  # 7000个romantic的句子
dataset_fu = []  # 7000个funny的句子

raw_style_dict = json.load(open('../data/raw_style_dict.json', 'r'))
raw_not_found = []

with open(dataset_dir_ro, 'rb') as f:
    while True:
        line = f.readline()
        line = line.decode('utf-8', 'ignore')
        if not line:
            break
        else:
            raw = line.replace("\r\n", "").rstrip().lstrip()
            tokens = nltk.word_tokenize(raw)
            sentence = [word.lower() for word in tokens[:-1]]
            if tokens[-1] != '.':   # 有的句子句号没有被分开
                if tokens[-1][-1] != '.':
                    print("Unexpect!")
                    print(tokens)
                    input()
                sentence += [tokens[-1][:-1]]
            if raw in raw_style_dict:
                caption_style = raw_style_dict[raw]
                caption_style = [word.lower() for word in caption_style]
            else:
                print("not found")
                raw_not_found.append(raw)
                caption_style = []
            if caption_style == []:
                print("empty")
            sentence = [word.replace("-", "") for word in sentence]
            dataset_ro.append([raw, sentence, caption_style])

with open(dataset_dir_fu, 'rb') as f:
    while True:
        line = f.readline()
        line = line.decode('utf-8', 'ignore')
        if not line:
            break
        else:
            raw = line.replace("\r\n", "").rstrip().lstrip()
            tokens = nltk.word_tokenize(raw)
            sentence = [word.lower() for word in tokens[:-1]]
            if tokens[-1] != '.':   # 有的句子句号没有被分开
                if tokens[-1][-1] != '.':
                    print("Unexpect!")
                    print(tokens)
                    input()
                sentence += [tokens[-1][:-1]]
            if raw in raw_style_dict:
                caption_style = raw_style_dict[raw]
                caption_style = [word.lower() for word in caption_style]
            else:
                print("not found")
                raw_not_found.append(raw)
                caption_style = []
            if caption_style == []:
                print("empty")
            sentence = [word.replace("-", "") for word in sentence]
            dataset_fu.append([raw, sentence, caption_style])

print("Num of Romantic Sentences: "+str(len(dataset_ro)))
print("Num of Funny Sentences: "+str(len(dataset_fu)))
print("Raw not found: "+str(len(raw_not_found)))

# 打乱7000个，分为5600 train、400 val、1000test，统一保存在一个文件中
dataset_flickrstyle = []
id_list = list(range(7000))
random.shuffle(id_list)
for step, id in tqdm(enumerate(id_list)):  # 每个id保存两个item（两种风格）
    if step < 5600:
        split = "train"
    elif step < 6000:
        split = "val"
    else:
        split = "test"
    filename = flickrstyle_filename_data[id]
    raw_ro = dataset_ro[id][0]
    raw_fu = dataset_fu[id][0]
    caption_ro = dataset_ro[id][1]
    caption_fu = dataset_fu[id][1]
    caption_style_ro = dataset_ro[id][2]
    caption_style_fu = dataset_fu[id][2]
    item_ro = {"raw": raw_ro, "caption": caption_ro, "style": "romantic", "caption_style": caption_style_ro, "filename": filename, "split": split}
    item_fu = {"raw": raw_fu, "caption": caption_fu, "style": "funny", "caption_style": caption_style_fu, "filename": filename, "split": split}
    dataset_flickrstyle.append(item_ro)
    dataset_flickrstyle.append(item_fu)

print("Num of flickrstyle: "+str(len(dataset_flickrstyle)))

out_dir = '../data/dataset_flickrstyle.json'
with open(out_dir, 'w') as f:
    json.dump(dataset_flickrstyle, f)
