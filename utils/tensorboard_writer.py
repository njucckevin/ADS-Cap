import torch
import os

def write_scalar(writer, scalar_name, scalar, step):
    writer.add_scalar(scalar_name, scalar, step)

def write_metrics(writer, pycoco_ro, pycoco_fu, pycoco_pos, pycoco_neg, ppl_ro, ppl_fu, ppl_pos, ppl_neg, cls_ro, cls_fu, cls_pos, cls_neg, step):
    pycoco_list = ["Bleu_1", "Bleu_3", "METEOR", "CIDEr"]
    for item in pycoco_list:
        write_scalar(writer, item+"_ro", pycoco_ro[item], step)
    write_scalar(writer, 'ppl_ro', ppl_ro, step)
    write_scalar(writer, 'cls_ro', cls_ro, step)
    for item in pycoco_list:
        write_scalar(writer, item+"_fu", pycoco_fu[item], step)
    write_scalar(writer, 'ppl_fu', ppl_fu, step)
    write_scalar(writer, 'cls_fu', cls_fu, step)
    for item in pycoco_list:
        write_scalar(writer, item+"_pos", pycoco_pos[item], step)
    write_scalar(writer, 'ppl_pos', ppl_pos, step)
    write_scalar(writer, 'cls_pos', cls_pos, step)
    for item in pycoco_list:
        write_scalar(writer, item+"_neg", pycoco_neg[item], step)
    write_scalar(writer, 'ppl_neg', ppl_neg, step)
    write_scalar(writer, 'cls_neg', cls_neg, step)
