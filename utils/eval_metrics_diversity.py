# 用于测试图像风格化描述多样性的一些指标

from tqdm import tqdm
from nltk import ngrams
from pycocoevalcap.eval import COCOEvalCap
from utils.style_extract import style_extract
from models.CLS.cls_lstm import Cls_Classifier
import torch
import math
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def list_to_str(sentence_list):
    sentence = ''
    for item in sentence_list:
        sentence += item
        sentence += ' '
    return sentence

def eval_distinct(all_captions):
    """计算生成的一组句子中不同caption的占比"""
    num_samples = len(all_captions)
    num_sentences = len(all_captions[0])
    ratio = 0
    for step in tqdm(range(num_sentences)):
        sentences = [item[step] for item in all_captions]
        distinct_sentences = set(sentences)
        ratio += (len(distinct_sentences)/len(sentences))

    ratio_mean = ratio / num_sentences
    return ratio_mean

def eval_distinct_style(all_captions, style):
    """计算生成的一组句子中风格部分不同caption的占比"""
    num_samples = len(all_captions)
    num_sentences = len(all_captions[0])
    ratio = 0
    for step in tqdm(range(num_sentences)):
        sentences = [list_to_str(style_extract(item[step], style)) for item in all_captions]
        distinct_sentences = set(sentences)
        ratio += (len(distinct_sentences)/len(sentences))

    ratio_mean = ratio / num_sentences
    return ratio_mean


def eval_ngram_diversity(all_captions, n):
    """计算生成的一组句子中不同的n-garm的占所有n-gram的比例"""
    num_samples = len(all_captions)
    num_sentences = len(all_captions[0])
    ratio = 0
    for step in tqdm(range(num_sentences)):
        sentences = [item[step].split() for item in all_captions]
        all_ngram = []
        for item in sentences:
            if n == 1:
                all_ngram += item
            elif n == 2:
                all_ngram += ngrams(item, 2)
        distinct_ngram = set(all_ngram)
        ratio += (len(distinct_ngram)/len(all_ngram))

    ratio_mean = ratio / num_sentences
    return ratio_mean


def eval_ngram_diversity_style(all_captions, n, style):
    """计算生成的一组句子中不同的n-garm的占所有n-gram的比例"""
    num_samples = len(all_captions)
    num_sentences = len(all_captions[0])
    ratio = 0
    for step in tqdm(range(num_sentences)):
        sentences = [style_extract(item[step], style) for item in all_captions]
        all_ngram = []
        for item in sentences:
            if n == 1:
                all_ngram += item
            elif n == 2:
                all_ngram += ngrams(item, 2)
        distinct_ngram = set(all_ngram)
        if len(all_ngram) != 0:
            ratio += (len(distinct_ngram)/len(all_ngram))

    ratio_mean = ratio / num_sentences
    return ratio_mean


def eval_mBleu4(all_captions):
    # mBleu4：生成一组num_samples个描述，对于其中的每个句子，计算该句子以剩余num_samples-1个句子为参考时的Bleu4，得分越高说明句子间越相似
    num_samples = len(all_captions)
    num_sentences = len(all_captions[0])
    num_samples = num_samples if num_samples < 5 else 5  # 按照论文，一般选5个
    mBleu4 = 0
    for i in range(num_samples):
        res_data = {}
        for j, sentence in enumerate(all_captions[i]):
            refs = [{"image_id": j, "caption": sentence, "id": 0}]
            res_data[j] = refs

        ref_data = {}
        for step in range(num_sentences):
            refs = [{"image_id": step, "caption": all_captions[j][step], "id": step}
                    for j in range(num_samples) if j != i]
            for k, item in enumerate(refs):
                item["id"] = k
            ref_data[step] = refs
        cocoEval = COCOEvalCap('nothing', 'nothing')
        scores = cocoEval.evaluate_diy(ref_data, res_data)
        mBleu4 += scores["Bleu_4"]

    mBleu4_mean = mBleu4 / num_samples
    return mBleu4_mean

def eval_wordfreq(all_captions, style):
    entropy_sum_mean = 0
    ratio_mean = 0
    if style == 'pos' or style == 'neg':
        categories = ["man", "woman", "people", "girl", "dog", "cat"]
    else:
        categories = ["man", "woman", "people", "boy", "girl", "dog"]
    results = {item: [] for item in categories}
    for cate in categories:
        all_captions_cate = [sentence for sentence in all_captions if cate in sentence.split()]
        counter = {}
        style_phrases = []
        for sentence in tqdm(all_captions_cate):
            style_phrase = style_extract(sentence, style)
            style_phrases.append(list_to_str(style_phrase))
            for token in style_phrase:
                counter[token] = counter.get(token, 0)+1
        word_count = [(token, f) for token, f in counter.items()]
        total_num = sum([item[1] for item in word_count])
        word_count = [(token, f/total_num) for token, f in word_count]
        word_freq = sorted(word_count, key=lambda x: x[1], reverse=True)
        entropy = [math.log(f, 2)*f for token, f in word_freq]
        entropy_sum = round(-sum(entropy), 2)
        distinct_style_phrases = set(style_phrases)
        #with open('/home/chengkz/checkpoints/MultiStyle_IC/log/style_phrase_dict.json', 'w') as f:
        #    json.dump(list(distinct_style_phrases), f)
        ratio = round(len(distinct_style_phrases)/len(style_phrases), 2)
        entropy_sum_mean += entropy_sum
        ratio_mean += ratio
        print("Category: "+cate+'  '+str(entropy_sum)+'  '+str(ratio))
        results[cate] = [ratio, entropy_sum]
    return round(entropy_sum_mean / len(categories), 2), round(ratio_mean / len(categories), 2), results
