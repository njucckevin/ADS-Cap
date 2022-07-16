# 测试多样性（根据提供的路径，测试路径下指定结果的多样性，并生成csv文件）
import sys
import os
import json
from utils.eval_metrics_diversity import eval_distinct, eval_ngram_diversity, eval_mBleu4
from utils.eval_metrics_diversity import eval_distinct_style, eval_ngram_diversity_style, eval_wordfreq
import csv

def write_csv(path, results):
    with open(path, 'w', encoding='utf-8') as f:
        csv_writer = csv.writer(f)
        categories = ["", "man", "woman", "people", "boy", "girl", "dog", "cat", "mean"]
        styles = ["Romantic", "Humorous", "Positive", "Negative"]
        csv_writer.writerow(categories)
        for i, result in enumerate(results):
            row = []
            for key in categories:
                if key == "":
                    row.append(styles[i])
                elif key in result:
                    row.append(str(result[key][1])+'/'+str(result[key][0]))
                else:
                    row.append('/')
            csv_writer.writerow(row)
    f.close()


id = sys.argv[1]
log_dir = '/home/chengkz/checkpoints/MultiStyle_IC_v3/log/{}'
results_dir = log_dir.format(id)+'/generated'
step = sys.argv[2]
num = int(sys.argv[3])
is_style_phrase = sys.argv[4]
styles = ["ro", "fu", "pos", "neg"]

results_all = []
for style in styles:
    result_path = os.path.join(results_dir, style+'_test_'+str(step)+'.txt')
    if num != 1:    # 测试第二类多样性
        result_path = os.path.join(results_dir, style + '5_test_' + str(step) + '.txt')
        num_samples = num
        all_caption = [[] for _ in range(num_samples)]
        with open(result_path, 'rb') as f:
            i = 0
            total_num = 0
            while True:
                line = f.readline()
                line = line.decode('utf-8', 'ignore')
                if not line:
                    break
                all_caption[i % num_samples].append(line)
                i += 1
                total_num += 1
        print("Total num: "+str(total_num))
        if is_style_phrase == 'no':
            print("mBleu-4: "+str(eval_mBleu4(all_caption)))
            print("Distinct: "+str(eval_distinct(all_caption)))
            print("1-gram Diversity: "+str(eval_ngram_diversity(all_caption, 1)))
            print("2-gram Diversity: "+str(eval_ngram_diversity(all_caption, 2)))
        else:
            print("Distinct: "+str(eval_distinct_style(all_caption, style)))
            print("1-gram Diversity: "+str(eval_ngram_diversity_style(all_caption, 1, style)))
            print("2-gram Diversity: "+str(eval_ngram_diversity_style(all_caption, 2, style)))
    else:   # 测试第一类多样性
        all_caption = []
        with open(result_path, 'rb') as f:
            while True:
                line = f.readline()
                line = line.decode('utf-8', 'ignore')
                if not line:
                    break
                all_caption.append(line)
        print("Total num: "+str(len(all_caption)))
        entropy, ratio, results = eval_wordfreq(all_caption, style)
        results["mean"] = [ratio, entropy]
        results_all.append(results)
        print("Entropy: "+str(entropy))
        print("Distinct Ratio: "+str(ratio))

output_path = os.path.join(results_dir, str(step)+'_div1.csv')
write_csv(output_path, results_all)

