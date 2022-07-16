# encoding: utf-8
# 整理senticap的2360张图片
import json
import os
import shutil
# from tqdm import tqdm

senticap_data_dir = '/home/data_ti4_c/chengkz/data/Senticap_dataset/senticap_dataset.json'
senticap_data = json.load(open(senticap_data_dir, 'r'))

out_path = '/home/data_ti4_c/chengkz/data/SentiCap2360'
if not os.path.exists(out_path):
    os.makedirs(out_path)
cocoimg_path = '/home/data_ti4_c/chengkz/data/coco_dataset/val2014'
num = 0
for step, item in enumerate(senticap_data["images"]):
    img_dir = os.path.join(cocoimg_path, item["filename"])
    out_dir = os.path.join(out_path, item["filename"])
    if os.path.exists(img_dir):
        num += 1
        shutil.copy(img_dir, out_dir)
    else:
        print("img not found!")

print("total num: "+str(num))