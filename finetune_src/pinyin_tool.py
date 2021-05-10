#encoding:utf8
import sys
import numpy as np

class PinyinTool:
    def __init__(self, py_dict_path, py_vocab_path, py_or_sk='py'):
        self.zi_pinyin = self._load_pydict(py_dict_path)
        self.vocab = self._load_pyvocab(py_vocab_path)
        if 'py' in py_or_sk:
            self.ZM2ID = {':':1, 'a':2, 'c':3, 'b':4, 'e':5, 'd':6, 'g':7, 'f':8, 'i':9, 'h':10, 'k':11, 'j':12, 'm':13, 'l':14, 'o':15, 'n':16, 'q':17, 'p':18, 's':19, 'r':20, 'u':21, 't':22, 'w':23, 'v':24, 'y':25, 'x':26, 'z':27}
            self.PYLEN = 4
        else:
            self.ZM2ID = {'1': 1, '2':2, '3':3, '4':4, '5':5}
            self.PYLEN = 10

    def _load_pydict(self, fpath):
        ans = {}
        for line in open(fpath, encoding='utf-8'):
            line = line.strip()#.decode('utf8')
            tmps = line.split('\t')
            if len(tmps) != 2: continue
            ans[tmps[0]] = tmps[1]
        return ans


    def _load_pyvocab(self, fpath):
        ans = {'PAD': 0, 'UNK': 1}
        idx = 2
        for line in open(fpath, encoding='utf-8'):
            line = line.strip()#.decode('utf8')
            if len(line) < 1: continue
            ans[line] = idx
            idx += 1
        return ans

    def get_pinyin_id(self, zi_unicode):
        py = self.zi_pinyin.get(zi_unicode, None)
        if py is None:
            return self.vocab['UNK']
        return self.vocab.get(py, self.vocab['UNK'])
            
    def get_pyid2seq_matrix(self):
        ans = [[0] * self.PYLEN, [0] * self.PYLEN] #PAD, UNK
        rpyvcab = {v: k  for k, v in self.vocab.items()}
        for k in range(2, len(rpyvcab), 1):
            pystr = rpyvcab[k]
            seq = []
            for c in pystr:
                seq.append(self.ZM2ID[c])
            seq = [0] * self.PYLEN + seq
            seq = seq[-self.PYLEN:]
            ans.append(seq)
        return np.asarray(ans, dtype=np.int32) 

