CUDA_VISIBLE_DEVICES=1 python -u train_style_classifier_for_extraction.py \
                --do_train \
                --do_eval \
                --output_dir output/classifier/flickr7k/ \
                --style0_train_text_path data/flickr7k/humor_train.txt \
                --style1_train_text_path data/flickr7k/romantic_train.txt \
                --style0_val_text_path data/flickr7k/humor_val.txt \
                --style1_val_text_path data/flickr7k/romantic_val.txt \
                --log_path logs/train_style_classifier_for_extraction_flickr7k.log \
                --max_length 64 \
                --save_steps 0 \
                --logging_steps 0 \
                --batch_size 32 \
                --num_train_epochs 3 \
                --weight_decay 0.04680559213273095 \
                --learning_rate 1.6239780813448106e-05