#*-*encoding:utf8*-*
import random
import copy
import time
import numpy as np
import tokenization

class ConfusionSet:
    def __init__(self, tokenizer, in_file):
        self.tokenizer = tokenizer
        self.confusion = self._load_confusion(in_file)

    def _str2idstr(self, string):
        ids = [self.tokenizer.vocab.get(x, -1) for x in string]
        if min(ids) < 0:
            return None
        ids = ' '.join(map(str, ids))
        return ids
 
    def _load_confusion(self, in_file):
        pass

    def get_confusion_item_by_ids(self, token_id):
        confu = self.confusion.get(token_id, None)
        if confu is None:
            return None
        return confu[random.randint(0,len(confu) - 1)]

    def get_confusion_item_by_unicode(self, key_unicode):
        if len(key_unicode) == 1:
            keyid = self.tokenizer.vocab.get(key_unicode, None)
        else:
            keyid = self._str2idstr(key_unicode)
        if keyid is None:
            return None
        confu = self.confusion.get(keyid, None)
        if confu is None:
            return None
        return confu[random.randint(0, len(confu) - 1)]

 
class PinyinConfusionSet(ConfusionSet):
    def _load_confusion(self, in_file):
        confusion_datas = {}
        for line in open(in_file, encoding='utf-8'):
            line = line.strip()#.decode('utf-8')
            tmps = line.split('\t')
            if len(tmps) != 2:
                continue
            key = tmps[0]
            values = tmps[1].split()
            if len(key) != 1:
                continue
            all_ids = set()
            keyid = self.tokenizer.vocab.get(key, None)
            if keyid is None:
                continue
            for k in values:
                if self.tokenizer.vocab.get(k, None) is not None:
                    all_ids.add(self.tokenizer.vocab[k])
            all_ids = list(all_ids)
            if len(all_ids) > 0:
                confusion_datas[keyid] = all_ids
        return confusion_datas

class StrokeConfusionSet(ConfusionSet):
    def _load_confusion(self, in_file):
        confusion_datas = {}
        for line in open(in_file, encoding='utf-8'):
            line = line.strip()#.decode('utf-8')
            tmps = line.split(',')
            if len(tmps) < 2:
                continue
            values = tmps
            all_ids = set()
            for k in values:
                if k in self.tokenizer.vocab:
                    all_ids.add(self.tokenizer.vocab[k])
            all_ids = list(all_ids)
            for k in all_ids:
                confusion_datas[k] = all_ids
        return confusion_datas



class Mask(object):
    def __init__(self, same_py_confusion, simi_py_confusion, sk_confusion, tokenid2pyid, tokenid2skid):
        self.same_py_confusion = same_py_confusion
        self.simi_py_confusion = simi_py_confusion
        self.sk_confusion = sk_confusion
        self.tokenid_pyid = tokenid2pyid
        self.tokenid_skid = tokenid2skid
        self.config = {'same_py': 0.3, 'simi_py': 0.3, 'stroke': 0.15, 'random': 0.1, 'keep': 0.15, 'global_rate': 0.15}
        self.same_py_thr = self.config['same_py'] 
        self.simi_py_thr = self.config['same_py'] + self.config['simi_py']
        self.stroke_thr = self.config['same_py'] + self.config['simi_py'] + self.config['stroke']
        self.random_thr = self.config['same_py'] + self.config['simi_py'] + self.config['stroke'] + self.config['random']
        self.keep_thr = self.config['same_py'] + self.config['simi_py'] + self.config['stroke'] + self.config['random'] + self.config['keep']
        self.invalid_ids = set([self.same_py_confusion.tokenizer.vocab.get('UNK'),
                               self.same_py_confusion.tokenizer.vocab.get('[CLS]'),
                               self.same_py_confusion.tokenizer.vocab.get('[SEP]'),
                               self.same_py_confusion.tokenizer.vocab.get('[UNK]')])

        self.all_token_ids = [int(x) for x in self.same_py_confusion.tokenizer.vocab.values()]
        self.n_all_token_ids = len(self.all_token_ids) - 1

    def get_mask_method(self):
        prob = random.random()
        if prob <= self.same_py_thr:
            return 'pinyin'
        elif prob <= self.simi_py_thr:
            return 'jinyin'
        elif prob <= self.stroke_thr:
            return 'stroke'
        elif prob <= self.random_thr:
            return 'random'
        elif prob <= self.keep_thr:
            return 'keep'
        return 'pinyin'

    def mask_process(self, input_sample):
        valid_ids = [idx for (idx, v) in enumerate(input_sample) if v not in self.invalid_ids]
        masked_sample = copy.deepcopy(list(input_sample))
        seq_len = len(masked_sample)
        masked_flgs = [0] * seq_len
        n_masked = int(len(valid_ids) * self.config['global_rate'])
        if n_masked < 1:
            n_masked = 1
        random.shuffle(valid_ids)
        for pos in valid_ids[:n_masked]:
            method = self.get_mask_method()
            if method == 'pinyin':
                new_c = self.same_py_confusion.get_confusion_item_by_ids(input_sample[pos])
                if new_c is not None:
                    masked_sample[pos] = new_c
                    masked_flgs[pos] = 1
            elif method == 'jinyin':
                new_c = self.simi_py_confusion.get_confusion_item_by_ids(input_sample[pos])
                if new_c is not None:
                    masked_sample[pos] = new_c
                    masked_flgs[pos] = 1
            elif method == 'stroke':
                new_c = self.sk_confusion.get_confusion_item_by_ids(input_sample[pos]) 
                if new_c is not None:
                    masked_sample[pos] = new_c
                    masked_flgs[pos] = 1
            elif method == 'random':
                new_c = self.all_token_ids[random.randint(0, self.n_all_token_ids)]
                if new_c is not None:
                    masked_sample[pos] = new_c
                    masked_flgs[pos] = 1
            elif method == 'keep': 
                masked_flgs[pos] = 1
        masked_py_ids = [self.tokenid_pyid.get(x, 1) for x in masked_sample]  
        masked_sk_ids = [self.tokenid_skid.get(x, 1) for x in masked_sample] 
        return np.asarray(masked_sample, dtype=np.int32), np.asarray(masked_flgs, dtype=np.int32), np.asarray(masked_py_ids, dtype=np.int32), np.asarray(masked_sk_ids, dtype=np.int32)

