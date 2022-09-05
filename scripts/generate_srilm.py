# 将包含句子的json文件转化为txt格式以便srilm处理，后续生成语言模型以计算forward ppl和backward ppl
import json

data_dir = '../data/dataset_flickrstyle.json'
dataset = json.load(open(data_dir, 'r'))
out_dir = '../PPL/text_srilm_fu.txt'

num = 0
with open(out_dir, 'w') as f:
    for item in dataset:
        if item["style"] == "funny":
            # for sentence_list in item["captions"]:
            num += 1
            sentence_list = item["caption"]
            sentence = ''
            for word in sentence_list:
                sentence += word
                sentence += ' '
            f.writelines(sentence+'\n')
print("Total sen: "+str(num))

f.close()