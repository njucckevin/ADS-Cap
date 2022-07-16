import torch
import os
import pickle
import json
from models.CLS.cls_lstm import Cls_Classifier
from models.Discriminator.discriminator_lstm import Discriminator_lstm
from data_load import data_load
from tqdm import tqdm
import random
from pycocoevalcap.eval import COCOEvalCap

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_sen_stylelabel(config, model, step, mode):
    """使用当前模型生成romantic和funny的句子并保存成json和txt格式以便后续计算B、M、C以及ppl和cls"""
    print("Generating sentence...")

    with open(config.vocab, 'rb') as f:
        vocab = pickle.load(f)
    log_path = config.log_dir.format(config.id)
    result_dir = os.path.join(log_path, 'generated')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    out_txt_dir_ro = result_dir+'/ro_'+mode+'_'+str(step)+'.txt'
    out_txt_dir_fu = result_dir+'/fu_'+mode+'_'+str(step)+'.txt'
    out_json_dir_ro = result_dir+'/ro_'+mode+'_'+str(step)+'.json'
    out_json_dir_fu = result_dir+'/fu_'+mode+'_'+str(step)+'.json'
    out_txt_dir_pos = result_dir+'/pos_'+mode+'_'+str(step)+'.txt'
    out_txt_dir_neg = result_dir+'/neg_'+mode+'_'+str(step)+'.txt'
    out_json_dir_pos = result_dir+'/pos_'+mode+'_'+str(step)+'.json'
    out_json_dir_neg = result_dir+'/neg_'+mode+'_'+str(step)+'.json'

    # 生成两种风格的
    if mode == 'val':
        items = [{'data_dir': config.val_ro, 'dim': 0, 'out_txt': out_txt_dir_ro, 'out_json': out_json_dir_ro},
                 {'data_dir': config.val_fu, 'dim': 1, 'out_txt': out_txt_dir_fu, 'out_json': out_json_dir_fu},
                 {'data_dir': config.val_pos, 'dim': 2, 'out_txt': out_txt_dir_pos, 'out_json': out_json_dir_pos},
                 {'data_dir': config.val_neg, 'dim': 3, 'out_txt': out_txt_dir_neg, 'out_json': out_json_dir_neg}]
    elif mode == 'test':
        items = [{'data_dir': config.test_ro, 'dim': 0, 'out_txt': out_txt_dir_ro, 'out_json': out_json_dir_ro},
                 {'data_dir': config.test_fu, 'dim': 1, 'out_txt': out_txt_dir_fu, 'out_json': out_json_dir_fu},
                 {'data_dir': config.test_pos, 'dim': 2, 'out_txt': out_txt_dir_pos, 'out_json': out_json_dir_pos},
                 {'data_dir': config.test_neg, 'dim': 3, 'out_txt': out_txt_dir_neg, 'out_json': out_json_dir_neg}
                 ]

    for item in items:
        val_loader = data_load(config, 'val', item['data_dir'])
        model.eval()
        cocoeval_out = {}
        num_generated = 1
        total_num = 0
        with open(item['out_txt'], 'w') as f:
            for i, (obj, obj_num, feat) in tqdm(enumerate(val_loader)):
                obj = obj.to(device)
                obj_num = obj_num.to(device)
                obj_num = obj_num + 2
                feat = feat.to(device)
                res_vec = model.resnet_feat_encoder(feat)
                obj_vec = model.objword_feat_encoder(obj)
                feat_vec = res_vec

                num = 0
                num_try = 0
                num_sample = 0
                while True:

                    style_label = torch.tensor([item["dim"]]).to(device)
                    # style_label = torch.tensor([4]).to(device)
                    sentence_id = model.generate_onesample(feat_vec, style_label)
                    # sentence_id, sentences = model.generate_beamsearch(feat_vec, style_label)

                    sentence = vocab.idList_to_sent(sentence_id)
                    f.writelines(sentence + '\n')

                    refs = []
                    ref = {}
                    ref['image_id'] = i
                    ref['caption'] = sentence
                    ref['id'] = i
                    refs.append(ref)
                    cocoeval_out[i] = refs

                    num += 1
                    total_num += 1

                    if num == num_generated:
                        break

        f.close()
        print("Total generated sentences: " + str(total_num))
        with open(item['out_json'], 'w') as f:
            json.dump(cocoeval_out, f)


def generate_sen_stylelabel_reval(config, model, step, mode):
    """使用当前模型生成romantic和funny的句子并保存成json和txt格式以便后续计算B、M、C以及ppl和cls"""
    print("Generating sentence...")

    with open(config.vocab, 'rb') as f:
        vocab = pickle.load(f)
    log_path = config.log_dir.format(config.id)
    result_dir = os.path.join(log_path, 'generated')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    out_txt_dir_ro = result_dir+'/ro_'+mode+'_'+str(step)+'.txt'
    out_txt_dir_fu = result_dir+'/fu_'+mode+'_'+str(step)+'.txt'
    out_json_dir_ro = result_dir+'/ro_'+mode+'_'+str(step)+'.json'
    out_json_dir_fu = result_dir+'/fu_'+mode+'_'+str(step)+'.json'
    out_txt_dir_pos = result_dir+'/pos_'+mode+'_'+str(step)+'.txt'
    out_txt_dir_neg = result_dir+'/neg_'+mode+'_'+str(step)+'.txt'
    out_json_dir_pos = result_dir+'/pos_'+mode+'_'+str(step)+'.json'
    out_json_dir_neg = result_dir+'/neg_'+mode+'_'+str(step)+'.json'

    # 生成两种风格的
    if mode == 'val':
        items = [{'data_dir': config.val_ro, 'dim': 0, 'out_txt': out_txt_dir_ro, 'out_json': out_json_dir_ro},
                 {'data_dir': config.val_fu, 'dim': 1, 'out_txt': out_txt_dir_fu, 'out_json': out_json_dir_fu},
                 {'data_dir': config.val_pos, 'dim': 2, 'out_txt': out_txt_dir_pos, 'out_json': out_json_dir_pos},
                 {'data_dir': config.val_neg, 'dim': 3, 'out_txt': out_txt_dir_neg, 'out_json': out_json_dir_neg}]
    elif mode == 'test':
        items = [{'data_dir': config.test_ro, 'dim': 0, 'out_txt': out_txt_dir_ro, 'out_json': out_json_dir_ro},
                 {'data_dir': config.test_fu, 'dim': 1, 'out_txt': out_txt_dir_fu, 'out_json': out_json_dir_fu},
                 {'data_dir': config.test_pos, 'dim': 2, 'out_txt': out_txt_dir_pos, 'out_json': out_json_dir_pos},
                 {'data_dir': config.test_neg, 'dim': 3, 'out_txt': out_txt_dir_neg, 'out_json': out_json_dir_neg}
                 ]

    model_d = Discriminator_lstm(config).to(device)
    model_d_path = config.recheck_model_path
    model_d.load_state_dict(torch.load(model_d_path))
    model_d.eval()
    softmax = torch.nn.Softmax(dim=0)

    for item in items:
        val_loader = data_load(config, 'val', item['data_dir'])
        model.eval()
        cocoeval_out = {}
        num_generated = 1
        total_num = 0

        with open(item['out_txt'], 'w') as f:
            for i, (obj, obj_num, feat) in tqdm(enumerate(val_loader)):
                obj = obj.to(device)
                obj_num = obj_num.to(device)
                obj_num = obj_num + 2
                feat = feat.to(device)
                res_vec = model.resnet_feat_encoder(feat)
                obj_vec = model.objword_feat_encoder(obj)
                feat_vec = res_vec

                num = 0
                while True:

                    style_label = torch.tensor([item["dim"]]).to(device)
                    sentence_id, sentences = model.generate_beamsearch(feat_vec, style_label)

                    found = False
                    for sentence_id in sentences:
                        sentence_id = torch.Tensor(sentence_id)
                        length = 0
                        for id in sentence_id:
                            if int(id) == 1:
                                continue
                            elif int(id) == 2 or int(id) == 0:
                                break
                            else:
                                length += 1
                        length = length + 2
                        cap = sentence_id.long().to(device).unsqueeze(0)
                        if length > cap.size(1):
                            length = cap.size(1)
                        cap_len = torch.tensor([length]).to(device)
                        pred = model_d(cap, cap_len)
                        pred_pro = softmax(pred[0])
                        if float(pred_pro[1]) > 0.9:
                            found = True
                            break

                    if found == False:
                        sentence_id = sentences[0]

                    sentence = vocab.idList_to_sent(sentence_id)
                    f.writelines(sentence + '\n')

                    refs = []
                    ref = {}
                    ref['image_id'] = i
                    ref['caption'] = sentence
                    ref['id'] = i
                    refs.append(ref)
                    cocoeval_out[i] = refs

                    num += 1
                    total_num += 1

                    if num == num_generated:
                        break

        f.close()
        print("Total generated sentences: " + str(total_num))
        with open(item['out_json'], 'w') as f:
            json.dump(cocoeval_out, f)


def generate_sen_stylelabel_test(config, model, step, mode):
    """使用当前模型生成romantic和funny的句子并保存成json和txt格式以便后续计算B、M、C以及ppl和cls"""
    print("Generating sentence...")

    with open(config.vocab, 'rb') as f:
        vocab = pickle.load(f)
    log_path = config.log_dir.format(config.id)
    result_dir = os.path.join(log_path, 'generated')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    out_txt_dir_ro = result_dir+'/ro5_'+mode+'_'+str(step)+'.txt'
    out_txt_dir_fu = result_dir+'/fu5_'+mode+'_'+str(step)+'.txt'
    out_json_dir_ro = result_dir+'/ro5_'+mode+'_'+str(step)+'.json'
    out_json_dir_fu = result_dir+'/fu5_'+mode+'_'+str(step)+'.json'
    out_txt_dir_pos = result_dir+'/pos5_'+mode+'_'+str(step)+'.txt'
    out_txt_dir_neg = result_dir+'/neg5_'+mode+'_'+str(step)+'.txt'
    out_json_dir_pos = result_dir+'/pos5_'+mode+'_'+str(step)+'.json'
    out_json_dir_neg = result_dir+'/neg5_'+mode+'_'+str(step)+'.json'

    # 生成两种风格的
    if mode == 'val':
        items = [{'data_dir': config.val_ro, 'dim': 0, 'out_txt': out_txt_dir_ro, 'out_json': out_json_dir_ro},
                 {'data_dir': config.val_fu, 'dim': 1, 'out_txt': out_txt_dir_fu, 'out_json': out_json_dir_fu},
                 {'data_dir': config.val_pos, 'dim': 2, 'out_txt': out_txt_dir_pos, 'out_json': out_json_dir_pos},
                 {'data_dir': config.val_neg, 'dim': 3, 'out_txt': out_txt_dir_neg, 'out_json': out_json_dir_neg}]
    elif mode == 'test':
        items = [{'data_dir': config.test_ro, 'dim': 0, 'out_txt': out_txt_dir_ro, 'out_json': out_json_dir_ro},
                 {'data_dir': config.test_fu, 'dim': 1, 'out_txt': out_txt_dir_fu, 'out_json': out_json_dir_fu},
                 {'data_dir': config.test_pos, 'dim': 2, 'out_txt': out_txt_dir_pos, 'out_json': out_json_dir_pos},
                 {'data_dir': config.test_neg, 'dim': 3, 'out_txt': out_txt_dir_neg, 'out_json': out_json_dir_neg}
                 ]

    model_d = Discriminator_lstm(config).to(device)
    model_d_path = config.recheck_model_path
    model_d.load_state_dict(torch.load(model_d_path))
    model_d.eval()
    softmax = torch.nn.Softmax(dim=0)

    for item in items:
        val_loader = data_load(config, 'val', item['data_dir'])
        model.eval()
        cocoeval_out = {}
        num_generated = 1
        total_num = 0

        with open(item['out_txt'], 'w') as f:
            for i, (obj, obj_num, feat) in tqdm(enumerate(val_loader)):
                obj = obj.to(device)
                obj_num = obj_num.to(device)
                obj_num = obj_num + 2
                feat = feat.to(device)
                res_vec = model.resnet_feat_encoder(feat)
                obj_vec = model.objword_feat_encoder(obj)
                feat_vec = res_vec

                num = 0
                while True:

                    style_label = torch.tensor([item["dim"]]).to(device)
                    sentence_id, sentences = model.generate_beamsearch(feat_vec, style_label)

                    sentence_5 = []
                    for sentence_id in sentences:
                        sentence_id = torch.Tensor(sentence_id)
                        length = 0
                        for id in sentence_id:
                            if int(id) == 1:
                                continue
                            elif int(id) == 2 or int(id) == 0:
                                break
                            else:
                                length += 1
                        length = length + 2
                        cap = sentence_id.long().to(device).unsqueeze(0)
                        if length > cap.size(1):
                            length = cap.size(1)
                        cap_len = torch.tensor([length]).to(device)
                        pred = model_d(cap, cap_len)
                        pred_pro = softmax(pred[0])
                        if float(pred_pro[1]) > 0.9:
                            sentence_5.append(sentence_id)
                            if len(sentence_5) == 5:
                                break

                    while len(sentence_5) < 5:
                        sentence_id = sentences[random.randint(0, len(sentences)-1)]
                        if sentence_id not in sentence_5:
                            sentence_5.append(sentence_id)

                    for sentence_id in sentence_5:
                        sentence = vocab.idList_to_sent(sentence_id)
                        f.writelines(sentence + '\n')

                    refs = []
                    ref = {}
                    ref['image_id'] = i
                    ref['caption'] = sentence
                    ref['id'] = i
                    refs.append(ref)
                    cocoeval_out[i] = refs

                    num += 1
                    total_num += 1

                    if num == num_generated:
                        break

        f.close()
        print("Total generated sentences: " + str(total_num))
        with open(item['out_json'], 'w') as f:
            json.dump(cocoeval_out, f)


def eval_pycoco(config, step, mode):
    print("Calculating pycoco...")
    ref_dir_ro = './data/pycocoref_ro_'+mode+'.json'
    ref_dir_fu = './data/pycocoref_fu_'+mode+'.json'
    ref_data_ro = json.load(open(ref_dir_ro, 'r'))
    ref_data_fu = json.load(open(ref_dir_fu, 'r'))
    ref_dir_pos = './data/pycocoref_pos_'+mode+'.json'
    ref_dir_neg = './data/pycocoref_neg_'+mode+'.json'
    ref_data_pos = json.load(open(ref_dir_pos, 'r'))
    ref_data_neg = json.load(open(ref_dir_neg, 'r'))

    log_path = config.log_dir.format(config.id)
    result_dir = os.path.join(log_path, 'generated')

    res_dir_ro = result_dir+'/ro_'+mode+'_'+str(step)+'.json'
    res_dir_fu = result_dir+'/fu_'+mode+'_'+str(step)+'.json'
    res_data_ro = json.load(open(res_dir_ro, 'r'))
    res_data_fu = json.load(open(res_dir_fu, 'r'))
    res_dir_pos = result_dir+'/pos_'+mode+'_'+str(step)+'.json'
    res_dir_neg = result_dir+'/neg_'+mode+'_'+str(step)+'.json'
    res_data_pos = json.load(open(res_dir_pos, 'r'))
    res_data_neg = json.load(open(res_dir_neg, 'r'))

    cocoEval = COCOEvalCap('nothing', 'nothing')
    pycoco_ro = cocoEval.evaluate_diy(ref_data_ro, res_data_ro)
    pycoco_fu = cocoEval.evaluate_diy(ref_data_fu, res_data_fu)
    pycoco_pos = cocoEval.evaluate_diy(ref_data_pos, res_data_pos)
    pycoco_neg = cocoEval.evaluate_diy(ref_data_neg, res_data_neg)

    return pycoco_ro, pycoco_fu, pycoco_pos, pycoco_neg  # 返回两个字典 key为["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR", "ROUGE_L", "CIDEr"]


def eval_ppl(config, step, mode):
    print("Calculating ppl...")
    def read_ppl(result_dir):
        with open(result_dir, 'rb') as f:
            while True:
                line = f.readline()
                line = line.decode('utf-8')
                if not line:
                    break
                last_line = line
        tokens = last_line.split()
        ppl = float(tokens[-3])
        # ppl1 = float(tokens[-1])
        return ppl

    log_path = config.log_dir.format(config.id)
    result_dir = os.path.join(log_path, 'generated')

    # 用于计算的txt文本（两种风格）
    out_txt_dir_ro = result_dir+'/ro_'+mode+'_'+str(step)+'.txt'
    out_txt_dir_fu = result_dir+'/fu_'+mode+'_'+str(step)+'.txt'
    out_txt_dir_pos = result_dir+'/pos_'+mode+'_'+str(step)+'.txt'
    out_txt_dir_neg = result_dir+'/neg_'+mode+'_'+str(step)+'.txt'
    # 保存ppl结果的路径
    ppl_ro_path = result_dir + '/ppl_ro_' + str(step)
    ppl_fu_path = result_dir + '/ppl_fu_' + str(step)
    ppl_pos_path = result_dir + '/ppl_pos_' + str(step)
    ppl_neg_path = result_dir + '/ppl_neg_' + str(step)
    # 计算ppl
    os.system('ngram -ppl ' + out_txt_dir_ro + ' -order 3 -lm ./PPL/LM_ro'+' > ' + ppl_ro_path)  # 计算ppl
    os.system('ngram -ppl ' + out_txt_dir_fu + ' -order 3 -lm ./PPL/LM_fu'+' > ' + ppl_fu_path)  # 计算ppl
    os.system('ngram -ppl ' + out_txt_dir_pos + ' -order 3 -lm ./PPL/LM_pos'+' > ' + ppl_pos_path)  # 计算ppl
    os.system('ngram -ppl ' + out_txt_dir_neg + ' -order 3 -lm ./PPL/LM_neg'+' > ' + ppl_neg_path)  # 计算ppl

    ppl_ro = read_ppl(ppl_ro_path)
    ppl_fu = read_ppl(ppl_fu_path)
    ppl_pos = read_ppl(ppl_pos_path)
    ppl_neg = read_ppl(ppl_neg_path)

    return ppl_ro, ppl_fu, ppl_pos, ppl_neg


def eval_cls(config, step, mode):
    print("Calculating cls...")
    def cal_cls(res_data, style):

        model = Cls_Classifier().to(device)
        best_model_path = './models/CLS/model_'+str(style)+'.pt'
        model.load_state_dict(torch.load(best_model_path))
        model.eval()

        num = 0
        for i in range(len(res_data)):
            sentence = res_data[str(i)][0]['caption']
            sentence_list = sentence.split(' ')
            style_pred = model(sentence_list)
            pred_id = style_pred[0].argmax()
            if int(pred_id) == 1:
                num += 1

        cls = num / len(res_data)
        return cls

    log_path = config.log_dir.format(config.id)
    result_dir = os.path.join(log_path, 'generated')

    res_dir_ro = result_dir+'/ro_'+mode+'_'+str(step)+'.json'
    res_dir_fu = result_dir+'/fu_'+mode+'_'+str(step)+'.json'
    res_data_ro = json.load(open(res_dir_ro, 'r'))
    res_data_fu = json.load(open(res_dir_fu, 'r'))
    res_dir_pos = result_dir+'/pos_'+mode+'_'+str(step)+'.json'
    res_dir_neg = result_dir+'/neg_'+mode+'_'+str(step)+'.json'
    res_data_pos = json.load(open(res_dir_pos, 'r'))
    res_data_neg = json.load(open(res_dir_neg, 'r'))

    cls_ro = cal_cls(res_data_ro, 'ro')
    cls_fu = cal_cls(res_data_fu, 'fu')
    cls_pos = cal_cls(res_data_pos, 'pos')
    cls_neg = cal_cls(res_data_neg, 'neg')

    return cls_ro, cls_fu, cls_pos, cls_neg





