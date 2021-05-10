#-*-coding:utf8-*-
import sys, os
import numpy as np
import tensorflow as tf
import modeling
import optimization
import time

os.environ["PYTHONIOENCODING"] = "utf-8"
tf.logging.set_verbosity(tf.logging.ERROR)

def score_f(ans, print_flg=False, only_check=False, out_dir=''):
    fout = open('%s/pred.txt' % out_dir, 'w', encoding="utf-8")
    total_gold_err, total_pred_err, right_pred_err = 0, 0, 0
    check_right_pred_err = 0
    inputs, golds, preds = ans
    assert len(inputs) == len(golds)
    assert len(golds) == len(preds)
    for ori, god, prd in zip(inputs, golds, preds):
        ori_txt = str(ori)
        god_txt = str(god) #''.join(list(map(str, god)))
        prd_txt = str(prd) #''.join(list(map(str, prd)))
        if print_flg is True:
            print(ori_txt, '\t', god_txt, '\t', prd_txt)
        if 'UNK' in ori_txt:
            continue
        if ori_txt == god_txt and ori_txt == prd_txt:
            continue
        if prd_txt != god_txt:
            fout.writelines('%s\t%s\t%s\n' % (ori_txt, god_txt, prd_txt)) 
        if ori != god:
            total_gold_err += 1
        if prd != ori:
            total_pred_err += 1
        if (ori != god) and (prd != ori):
            check_right_pred_err += 1
            if god == prd:
                right_pred_err += 1
    fout.close()

    #check p, r, f
    p = 1. * check_right_pred_err / (total_pred_err + 0.001)
    r = 1. * check_right_pred_err / (total_gold_err + 0.001)
    f = 2 * p * r / (p + r +  1e-13)
    print('token check: p=%.3f, r=%.3f, f=%.3f' % (p, r, f))
    if only_check is True:
        return p, r, f

    #correction p, r, f
    #p = 1. * right_pred_err / (total_pred_err + 0.001)
    pc = 1. * right_pred_err / (check_right_pred_err + 0.001)
    rc = 1. * right_pred_err / (total_gold_err + 0.001)
    fc = 2 * pc * rc / (pc + rc + 1e-13) 
    print('token correction: p=%.3f, r=%.3f, f=%.3f' % (pc, rc, fc))
    return p, r, f


def score_f_py(ans_py, ans_zi, out_dir, print_flg=False, only_check=False):
    fout = open('%s/pred_py.txt' % out_dir, 'w', encoding="utf-8")
    total_gold_err, total_pred_err, right_pred_err = 0, 0, 0
    check_right_pred_err = 0
    inputs, golds, preds = ans_py
    inputs_z, golds_z, preds_z = ans_zi
    assert len(inputs) == len(golds)
    assert len(golds) == len(preds)
    assert len(inputs_z) == len(golds_z)

    index = -1
    total_len = len(inputs_z)
    for ori, god, prd in zip(inputs_z, golds_z, preds_z):
        index += 1
        ori_txt = str(ori)
        god_txt = str(god) #''.join(list(map(str, god)))
        prd_txt = str(prd) #''.join(list(map(str, prd)))
        if print_flg is True:
            print(ori_txt, '\t', god_txt, '\t', prd_txt)
        if 'UNK' in ori_txt:
            continue
        ori_py, god_py, prd_py = str(inputs[index]), str(golds[index]), str(preds[index])
        if (ori_txt == god_txt and ori_txt == prd_txt and prd_py == ori_py):
            continue
        if (god_txt != prd_txt) or (prd_py != ori_py):
            start_idx = index - 5
            if start_idx < 0: start_idx = 0
            end_idx = index + 5
            if end_idx > total_len: end_idx = total_len
            for _idx in range(start_idx, end_idx, 1):
                fout.writelines('%s\t%s\t%s\t%s\t%s\t%s\n' % (inputs_z[_idx], golds_z[_idx], preds_z[_idx], inputs[_idx], golds[_idx], preds[_idx])) 
            fout.writelines('\n')
        if ori != god:
            total_gold_err += 1
        if (prd != ori) or (prd_py != ori_py):
            total_pred_err += 1
        
        if (ori != god) and ((prd != ori) or (prd_py != ori_py)):
            check_right_pred_err += 1
            if god_py == prd_py:
                right_pred_err += 1
    fout.close()

    #check p, r, f
    p = 1. * check_right_pred_err / (total_pred_err + 0.001)
    r = 1. * check_right_pred_err / (total_gold_err + 0.001)
    f = 2 * p * r / (p + r +  1e-13)
    print('token check: p=%.3f, r=%.3f, f=%.3f' % (p, r, f))
    if only_check is True:
        return p, r, f

    #correction p, r, f
    #p = 1. * right_pred_err / (total_pred_err + 0.001)
    pc = 1. * right_pred_err / (check_right_pred_err + 0.001)
    rc = 1. * right_pred_err / (total_gold_err + 0.001)
    fc = 2 * pc * rc / (pc + rc + 1e-13) 
    print('token correction: p=%.3f, r=%.3f, f=%.3f' % (pc, rc, fc))
    return p, r, f




def score_f_sent(inputs, golds, preds):
    assert len(inputs) == len(golds)
    assert len(golds) == len(preds)
    total_gold_err, total_pred_err, right_pred_err = 0, 0, 0
    check_right_pred_err = 0
    fout = open('sent_pred_result.txt', 'w', encoding='utf-8')
    for ori_tags, god_tags, prd_tags in zip(inputs, golds, preds):
        assert len(ori_tags) == len(god_tags)
        assert len(god_tags) == len(prd_tags)
        gold_errs = [idx for (idx, tk) in enumerate(god_tags) if tk != ori_tags[idx]]
        pred_errs = [idx for (idx, tk) in enumerate(prd_tags) if tk != ori_tags[idx]]
        if len(gold_errs) > 0 or len(pred_errs) > 0:
            fout.writelines('\n%s\n%s\n%s\n' % ('|'.join(ori_tags), '|'.join(god_tags),'|'.join(prd_tags)))
        if len(gold_errs) > 0:
            total_gold_err += 1
            fout.writelines('gold_err\n')
        if len(pred_errs) > 0:
            fout.writelines('check_err\n')
            total_pred_err += 1
            if gold_errs == pred_errs:
                check_right_pred_err += 1
                fout.writelines('check_right\n')
            if god_tags == prd_tags:
                right_pred_err += 1
                fout.writelines('correct_right\n')
    fout.close()
    p = 1. * check_right_pred_err / total_pred_err
    r = 1. * check_right_pred_err / total_gold_err
    f = 2 * p * r / (p + r + 1e-13)
    #print(total_gold_err, total_pred_err, right_pred_err, check_right_pred_err)
    print('sent check: p=%.3f, r=%.3f, f=%.3f' % (p, r, f))
    p = 1. * right_pred_err / total_pred_err
    r = 1. * right_pred_err / total_gold_err
    f = 2 * p * r / (p + r + 1e-13)
    print('sent correction: p=%.3f, r=%.3f, f=%.3f' % (p, r, f))
    return p, r, f
