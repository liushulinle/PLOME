#-*-coding:utf8-*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import random
from collections import namedtuple
import re
import numpy as np
import tensorflow as tf
import csv
import tokenization
from pinyin_tool import PinyinTool

DEBUG = False

InputExample = namedtuple('InputExample', ['tokens', 'labels'])
InputFeatures = namedtuple('InputFeature', ['input_ids', 'input_mask', 'segment_ids', 'stroke_ids', 'lmask', 'label_ids'])

def get_tfrecord_num(tf_file):
    num = 0
    for record in tf.python_io.tf_record_iterator(tf_file):
        num += 1
    return num

class DataProcessor:
    '''
    data format:
    sent1\tsent2
    '''
    def __init__(self, input_path, max_sen_len, vocab_file, out_dir, label_list=None, is_training=True):
        self.input_path = input_path
        self.max_sen_len = max_sen_len
        self.is_training = is_training
        self.dataset = None
        self.out_dir = out_dir
        self.tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=False)
        self.label_list = label_list
        if label_list is not None:
            self.label_map = {}
            for (i, label) in enumerate(self.label_list):
                self.label_map[label] = i
        else:
            self.label_map = self.tokenizer.vocab
            self.label_list = {}
            for key in self.tokenizer.vocab:
                self.label_list[self.tokenizer.vocab[key]] = key

        py_dict_path = './pinyin_data/zi_py.txt'
        py_vocab_path = './pinyin_data/py_vocab.txt'
        sk_dict_path = './stroke_data/zi_sk.txt'
        sk_vocab_path = './stroke_data/sk_vocab.txt'
 
        self.pytool = PinyinTool(py_dict_path=py_dict_path, py_vocab_path=py_vocab_path, py_or_sk='py')
        self.sktool = PinyinTool(py_dict_path=sk_dict_path, py_vocab_path=sk_vocab_path, py_or_sk='sk')

        self.PYID2SEQ = self.pytool.get_pyid2seq_matrix() 
        self.SKID2SEQ = self.sktool.get_pyid2seq_matrix()

        self.py_label_list = {v: k for k, v in self.pytool.vocab.items()}
 
        self.tokenid_pyid = {}
        self.tokenid_skid = {}
        for key in self.tokenizer.vocab:
            self.tokenid_pyid[self.tokenizer.vocab[key]] = self.pytool.get_pinyin_id(key)    
            self.tokenid_skid[self.tokenizer.vocab[key]] = self.sktool.get_pinyin_id(key)    
        if input_path is not None: 
            if is_training is True:
                self.tfrecord_path = os.path.join(out_dir, "train.tf_record")
            else:
                self.tfrecord_path = os.path.join(out_dir, "eval.tf_record")
                #os.remove(self.tfrecord_path)
            if os.path.exists(self.tfrecord_path) is False:
                self.file2features()
            else:
                self.num_examples = get_tfrecord_num(self.tfrecord_path)
    def get_zi_py_matrix(self):
        pysize = 430
        matrix = []
        for k in range(len(self.tokenizer.vocab)):
            matrix.append([0] * pysize)

        for key in self.tokenizer.vocab:
            tokenid = self.tokenizer.vocab[key]
            pyid = self.pytool.get_pinyin_id(key)
            matrix[tokenid][pyid] = 1.
        return np.asarray(matrix, dtype=np.float32) 
       
         
    def sample(self, text_unicode1, text_unicode2):
        segs1 = text_unicode1.strip().split(' ')
        segs2 = text_unicode2.strip().split(' ')
        tokens, labels = [], []
        if len(segs1) != len(segs2):
            return None
        for x, y in zip(segs1, segs2):
            tokens.append(x)
            labels.append(y)
        if len(tokens) < 2: return None
        return InputExample(tokens=tokens, labels=labels)

    def load_examples(self):
        '''sent1 \t sent2'''
        train_data = open(self.input_path, encoding="utf-8")
        instances = []
        n_line = 0
        for ins in train_data:
            n_line += 1
            if (DEBUG is True) and (n_line > 1000):
                break
            #ins = ins.decode('utf8')
            tmps = ins.strip().split('\t')
            if len(tmps) < 2: 
                continue
            ins = self.sample(tmps[0], tmps[1])
            if ins is not None:
                instances.append(ins)

        if self.is_training:
            random.seed = 666
            random.shuffle(instances)
        return instances

    def convert_single_example(self, ex_index, example):
        label_map = self.label_map
        tokens = example.tokens
        labels = example.labels
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens) > self.max_sen_len - 2:
            tokens = tokens[0:(self.max_sen_len - 2)]
            labels = labels[0:(self.max_sen_len - 2)]

        _tokens = []
        _labels = []
        _lmask = []
        segment_ids = []
        stroke_ids = []
        _tokens.append("[CLS]")
        _lmask.append(0)
        _labels.append(labels[0])
        segment_ids.append(0)
        stroke_ids.append(0)
        for token, label in zip(tokens, labels):
            _tokens.append(token)
            _labels.append(label)
            _lmask.append(1)
            segment_ids.append(self.pytool.get_pinyin_id(token))
            stroke_ids.append(self.sktool.get_pinyin_id(token))
        _tokens.append("[SEP]")
        segment_ids.append(0)
        stroke_ids.append(0)
        _labels.append(labels[0])
        _lmask.append(0)

        input_ids = self.tokenizer.convert_tokens_to_ids(_tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < self.max_sen_len:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            stroke_ids.append(0)
            _labels.append(labels[0])
            _lmask.append(0)

        assert len(input_ids) == self.max_sen_len
        assert len(input_mask) == self.max_sen_len
        assert len(segment_ids) == self.max_sen_len
        assert len(stroke_ids) == self.max_sen_len

        label_ids = [label_map.get(l, label_map['UNK']) for l in _labels]

        feature = InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            stroke_ids=stroke_ids,
            lmask=_lmask,
            label_ids=label_ids
            )
        return feature
    
    def get_label_list(self):
        return self.label_list
 
    def file2features(self):
        output_file = self.tfrecord_path
        if os.path.exists(output_file):
            os.remove(output_file)
        examples = self.load_examples()
        self.num_examples = len(examples)
        writer = tf.python_io.TFRecordWriter(output_file)
        for (ex_index, example) in enumerate(examples):
            if ex_index % 1000 == 0:
                tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

            feature = self.convert_single_example(ex_index, example)
            create_int_feature = lambda values: tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            features = collections.OrderedDict()
            features["input_ids"] = create_int_feature(feature.input_ids)
            features["input_mask"] = create_int_feature(feature.input_mask)
            features["segment_ids"] = create_int_feature(feature.segment_ids)
            features["stroke_ids"] = create_int_feature(feature.stroke_ids)
            features["lmask"] = create_int_feature(feature.lmask)
            features["label_ids"] = create_int_feature(feature.label_ids)

            tf_example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(tf_example.SerializeToString())
    
    def build_data_generator(self, batch_size):
        def _get_py_seq(token_seq):
            ans = []
            for t in list(token_seq):
                pyid = self.tokenid_pyid.get(t, 1)
                ans.append(pyid)
            ans = np.asarray(ans, dtype=np.int32)
            return ans

        def _decode_record(record):
            """Decodes a record to a TensorFlow example."""
            name_to_features = {
            "input_ids": tf.FixedLenFeature([self.max_sen_len], tf.int64),
            "input_mask": tf.FixedLenFeature([self.max_sen_len], tf.int64),
            "segment_ids": tf.FixedLenFeature([self.max_sen_len], tf.int64),
            "stroke_ids": tf.FixedLenFeature([self.max_sen_len], tf.int64),
            "lmask": tf.FixedLenFeature([self.max_sen_len], tf.int64),
            "label_ids": tf.FixedLenFeature([self.max_sen_len], tf.int64),
            }


            example = tf.parse_single_example(record, name_to_features)

            #int64 to int32
            for name in list(example.keys()):
                t = example[name]
                if t.dtype == tf.int64:
                    t = tf.to_int32(t)
                example[name] = t
            input_ids = example['input_ids']
            input_mask = example['input_mask']
            segment_ids = example['segment_ids']
            stroke_ids = example['stroke_ids']
            label_ids = example['label_ids']
            lmask = example['lmask']
            py_labels = tf.py_func(_get_py_seq, [label_ids], [tf.int32])

            return input_ids, input_mask, segment_ids, stroke_ids, lmask, label_ids, py_labels
        if self.dataset is not None:
            return self.dataset

        dataset = tf.data.TFRecordDataset(self.tfrecord_path)
        dataset = dataset.map(_decode_record, num_parallel_calls=10)
        if self.is_training:
            dataset = dataset.repeat().shuffle(buffer_size=100)
        dataset = dataset.batch(batch_size).prefetch(50)
        self.dataset = dataset
        return dataset

    def get_feature(self, u_input, u_output=None):
        if u_output is None:
            u_output = u_input
        instance = self.sample(u_input, u_output)
        feature = self.convert_single_example(0, instance)
        input_ids = feature.input_ids
        input_mask = feature.input_mask
        input_py_ids = feature.segment_ids
        input_sk_ids = feature.stroke_ids
        label_ids = feature.label_ids
        label_mask = feature.lmask
        return input_ids, input_mask, input_py_ids, input_sk_ids, label_ids, label_mask

        
        

