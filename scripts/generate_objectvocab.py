# 生成object word词表，来自VG数据集给出的1600个类别objects_vocab.txt
import json

def plural(word):
    if word.endswith('y'):
        return word[:-1] + 'ies'
    elif word[-1] in 'sx' or word[-2:] in ['sh', 'ch']:
        return word + 'es'
    elif word.endswith('an'):
        return word[:-2] + 'en'
    else:
        return word + 's'

def single(word):
    if word.endswith('s'):
        return word[:-1]
    elif word.endswith('en'):
        return word[:-2] + 'en'
    elif word.endswith('ies'):
        return word[:-3] + 'y'
    else:
        return word

data_dir = '../data/objects_vocab.txt'
out_dir = '../data/objects_vocab_plus.json'

dataset_objects = []
with open(data_dir, 'rb') as f:
    while True:
        line = f.readline().splitlines()
        if not line:
            break
        line = line[0]
        line = line.decode('utf-8', 'ignore')
        words = line.split()
        for word in words:
            dataset_objects.append(word)

# 目标是找到VG数据集中1600个类别对应各种【可能】的object words
dataset_objects_plus = []
for word in dataset_objects:
    dataset_objects_plus.append(word)  # 原型
    word_plural = plural(word)  # 复数
    dataset_objects_plus.append(word_plural)
    word_single = single(word)  # 单数
    dataset_objects_plus.append(word_single)

dataset_objects_plus = list(set(dataset_objects_plus))
print("Num of object word: "+str(len(dataset_objects_plus)))
with open(out_dir, 'w') as f:
    json.dump(dataset_objects_plus, f)
