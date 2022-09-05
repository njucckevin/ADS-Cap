# 使用Similar Scenes arouse Similar Emotions作者给出的代码进行style phrase提取
# 思路是使用将句子输入一个基于bert的风格判别器，然后把每个词的attention（某一个layer和head）作为这个词的重要性权重
# 为使提取更准确做了一点改进：对于pos和neg，按照权重依次删除单词直到风格发生改变；对于ro和fu，则按照原论文提取前25%的词作为style phrase
import torch
from transformers import (BertForSequenceClassification, BertTokenizer)
import os
# os.chdir('./utils')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载预训练好的基于bert的风格判别器
models_diff_style = {}
pretrain_bert_path = '../bert_style_classifier_ckpts/flickr7k'
tokenizer = BertTokenizer.from_pretrained(pretrain_bert_path, do_lower_case=True)
model = BertForSequenceClassification.from_pretrained(pretrain_bert_path).to(device)
models_diff_style["flickr7k"] = [model, tokenizer]
pretrain_bert_path = '../bert_style_classifier_ckpts/senticap'
tokenizer = BertTokenizer.from_pretrained(pretrain_bert_path, do_lower_case=True)
model = BertForSequenceClassification.from_pretrained(pretrain_bert_path).to(device)
models_diff_style["senticap"] = [model, tokenizer]

# 加载使用的layer和head
score_flickr7k, _, _, _ = torch.load('../bert_style_classifier_ckpts/flickr7k/save_every_head-flickr7k-val.pt')
models_diff_style["flickr7k"].append(score_flickr7k[0][0])
score_senticap, _, _, _ = torch.load('../bert_style_classifier_ckpts/senticap/save_every_head-senticap-val.pt')
models_diff_style["senticap"].append(score_senticap[0][0])

def style_extract_posneg(sentence, style):
    dataset = "flickr7k" if (style == "ro" or style == "fu") else "senticap"
    model, tokenizer = models_diff_style[dataset][0], models_diff_style[dataset][1]
    model.eval()
    best_layer, best_head = models_diff_style[dataset][2]
    encodings = tokenizer([sentence], truncation=True, padding=True, max_length=30)
    input_ids = torch.tensor(encodings['input_ids']).to(device)
    attention_mask = torch.tensor(encodings['attention_mask']).to(device)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])[:attention_mask.sum().item()]

    # 迭代式一个词一个词删除，直到风格发生改变
    style_phrase = []
    while True:
        outputs = model(input_ids, attention_mask=attention_mask, labels=torch.tensor(0 if style == 'fu' else 1).to(device),
                        output_attentions=True)

        if float(torch.softmax(outputs.logits, dim=1)[0][1 if style == 'pos' else 0]) < 0.98:
            break
        attentions = outputs.attentions[best_layer][:, best_head, :, :]
        _, topi = attentions[0][0].topk(attention_mask.sum().item())
        topi = topi.tolist()

        # 不考虑sep和cls标志，找到要删除的那个词的idx
        i = 0
        while topi[i] == 0 or (topi[i] == len(topi)-1):
            i += 1
        idx = topi[i]

        # 将berttokenizer中切分的词复原
        if '##' not in tokens[idx]:
            input_ids = input_ids[:, torch.arange(input_ids.size(1)) != idx]
            attention_mask = attention_mask[:, torch.arange(attention_mask.size(1)) != idx]
            style_phrase.append(tokens[idx])
            del(tokens[idx])
        else:
            # 将[j, k)之间的部分删除并整理为一个词
            j = idx
            while (j >= 0 and '##' in tokens[j]):
                j -= 1
            k = idx + 1
            while (k < len(tokens) and '##' in tokens[k]):
                k += 1
            aft_del_range = [i for i in range(input_ids.size(1)) if (i < j or k <= i)]
            input_ids = input_ids[:, aft_del_range]
            attention_mask = attention_mask[:, aft_del_range]
            style_word = ''
            for i in range(j, k):
                style_word += tokens[i]
            style_word = style_word.replace("##", "")
            style_phrase.append(style_word)
            del(tokens[j:k])


    return style_phrase


def style_extract_rofu(sentence, style):
    dataset = "flickr7k" if (style == "ro" or style == "fu") else "senticap"
    model, tokenizer = models_diff_style[dataset][0], models_diff_style[dataset][1]
    model.eval()
    best_layer, best_head = models_diff_style[dataset][2]
    encodings = tokenizer([sentence], truncation=True, padding=True, max_length=30)
    input_ids = torch.tensor(encodings['input_ids']).to(device)
    attention_mask = torch.tensor(encodings['attention_mask']).to(device)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])[:attention_mask.sum().item()]
    outputs = model(input_ids, attention_mask=attention_mask, labels=torch.tensor(0 if style == 'fu' else 1).to(device),
                    output_attentions=True)
    attentions = outputs.attentions[best_layer][:, best_head, :, :]
    _, topi = attentions[0][0].topk(torch.tensor(encodings['attention_mask']).sum().item())
    topi = topi.tolist()

    # 直接删除前25%的词作为style phrase
    rm_topi = topi[:int(len(topi) * 0.25)]
    rm_ids = set()
    for idx in rm_topi:
        if idx == 0 or (idx == len(topi)-1):
            continue
        rm_ids.add(idx)
        # 将berttokenizer中切分的词复原
        if '##' in tokens[idx]:
            j = idx
            while (j >= 0 and '##' in tokens[j]):
                rm_ids.add(j)
                j -= 1
            if j >= 0:
                rm_ids.add(j)
            j = idx + 1
            while (j < len(tokens) and '##' in tokens[j]):
                rm_ids.add(j)
                j += 1

    style_phrase = (" ".join([token for j, token in enumerate(tokens) if j in rm_ids]).replace(" ##", "")).split()
    return style_phrase


def style_extract(sentence, style):
    if style == 'ro' or style == 'fu':
        return style_extract_rofu(sentence, style)
    elif style == 'pos' or style == 'neg':
        return style_extract_posneg(sentence, style)

"""
style_phrases_sl = json.load(open('/Users/cckevin/Desktop/style_phrase_dict_sl.json', 'r'))
style_phrases = json.load(open('/Users/cckevin/Desktop/style_phrase_dict.json', 'r'))
num = 0
for item in style_phrases:
    if item not in style_phrases_sl:
        print(item)
        #input()
        num += 1
print("Num: "+str(num))
"""
"""
style = "ro"
sentence_list = ["two", "joyous", "children", "chase", "each", "other", "unselfconsciously", "on", "the", "sand"]
sentence = ' '.join(sentence_list)
# sentence = "a delicious piece of tasty food on a plate"
style_phrase = style_extract(sentence, style)
print(style_phrase)
"""
