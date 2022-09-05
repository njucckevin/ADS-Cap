# ADS-Cap: A Framework for Accurate and Diverse Stylized Captioning with Unpaired Stylistic Corpora

## Prepare data

Step1: prepare SentiCap img `generate_senticapimg.py`

Step2: generate resnet features for all used images `generate_resnet_feat.py`

Step3: prepare objects vocabulary using VG dataset's object labels `generate_objectvocab.py`

Step4: prepare data of FlcikrStyle and SentiCap `prepro_flickrstyledata.py` & `prepro_senticapdata.py`

Step5: construct train/val/test data `generate_dataset.py`

Step6: prepare for calculating PPL using SRILM `generate_srilm.py`, reference: https://blog.csdn.net/u011500062/article/details/50781101, https://ynuwm.github.io/2017/05/24/SRILM训练语言模型实战/, http://www.mamicode.com/info-detail-1944347.html

Step7: build vocab `build_vocab.py`

Step8: prepare json file for pycocoeval `generate_cocoeval.py`

## Training

Step1: pretrain on coco dataset `CUDA_VISIBLE_DEVICES=0 python train_cvae.py --id cvae_k0.03_s1.0 --kl_rate 0.03 --style_rate 1.0 --save_model_freq 20000`

Step2: finetune on stylized datasets `CUDA_VISIBLE_DEVICES=0 python train_cvae.py --id cvae_k0.03_s1.0_ft --kl_rate 0.03 --style_rate 1.0 --finetune True --pretrain_id cvae_k0.03_s1.0 --pretrain_step 80000 --batch_size 50 --lr 5e-5 --save_model_freq 2700`

## Evaluation
Step1: generate captions and calculate accuracy metrics `CUDA_VISIBLE_DEVICES=0 python test_cvae.py --id cvae_k0.03_s1.0_ft --step 108000`

Step2: calculate diversity metrics; diversity across image `python test_diversity.py cvae_k0.03_s1.0_ft 108000 1 no`, diversity for one image `python test_diversity.py cvae_k0.03_s1.0_ft 108000 2 yes`
