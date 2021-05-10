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
from mask import Mask, PinyinConfusionSet, StrokeConfusionSet
from pinyin_tool import PinyinTool
 
DEBUG = False

InputExample = namedtuple('InputExample', ['tokens'])
InputFeatures = namedtuple('InputFeature', ['input_ids', 'input_mask', 'segment_ids', 'lmask']) #segment_ids is for pinyin_ids

def get_tfrecord_num(tf_file):
    num = 0
    for record in tf.python_io.tf_record_iterator(tf_file):
        num += 1
        if num > 300000:
            num = 50000000
            break
    return num

class DataProcessor:
    def __init__(self, input_path, max_sen_len, vocab_file, out_dir, label_list=None, is_training=True):
        self.input_path = input_path
        self.max_sen_len = max_sen_len
        self.is_training = is_training
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
        self.pplen = len(self.sktool.ZM2ID)
        self.sklen = self.sktool.PYLEN

        self.PYID2SEQ = self.pytool.get_pyid2seq_matrix()
        self.SKID2SEQ = self.sktool.get_pyid2seq_matrix()

        tokenid_pyid = {}
        tokenid_skid = {}
        for key in self.tokenizer.vocab:
            tokenid_pyid[self.tokenizer.vocab[key]] = self.pytool.get_pinyin_id(key)
            tokenid_skid[self.tokenizer.vocab[key]] = self.sktool.get_pinyin_id(key)
        

        same_py_file = './confusions/same_pinyin.txt'
        simi_py_file = './confusions/simi_pinyin.txt'
        stroke_file = './confusions/same_stroke.txt'
        tokenizer = self.tokenizer
        pinyin = PinyinConfusionSet(tokenizer, same_py_file)
        jinyin = PinyinConfusionSet(tokenizer, simi_py_file)
        print('pinyin conf size:', len(pinyin.confusion))
        print('jinyin conf size:', len(jinyin.confusion))
        stroke = StrokeConfusionSet(tokenizer, stroke_file)
        self.masker = Mask(same_py_confusion=pinyin, simi_py_confusion=jinyin, sk_confusion=stroke, tokenid2pyid=tokenid_pyid, tokenid2skid=tokenid_skid)


        file_pattern = out_dir + '/*.tfrecord' 
        if input_path is not None: 
            if is_training is True:
                pass
            else:
                self.tfrecord_path = out_dir
            if is_training is False:
                self.file2features()
            else:
                self.TfrecordFile = tf.gfile.Glob(file_pattern)
                self.TfrecordFile = sorted(self.TfrecordFile)
                random.shuffle(self.TfrecordFile)
                print ('--'.join(self.TfrecordFile))
                self.num_examples = 50000000
   
    def sample(self, text_unicode):
        tokens = text_unicode.strip().split(' ')
        if len(tokens) < 2: return None
        return InputExample(tokens=tokens)

    def load_examples(self):
        '''sent'''
        train_data = open(self.input_path, encoding="utf-8")
        #train_data = open(self.input_path)
        instances = []
        n_line = 0
        for ins in train_data:
            #ins = ins.decode('utf8')
            n_line += 1
            if (DEBUG is True) and (n_line > 10000):
                break
            ins = self.sample(ins)
            if ins is not None:
                yield ins

    def convert_single_example(self, ex_index, example):
        label_map = self.label_map
        tokens = example.tokens
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens) > self.max_sen_len - 2:
            tokens = tokens[0:(self.max_sen_len - 2)]

        _tokens = []
        _lmask = []
        segment_ids = []
        _tokens.append("[CLS]")
        _lmask.append(0)
        segment_ids.append(0)
        for token in tokens:
            _tokens.append(token)
            _lmask.append(1)
            segment_ids.append(self.pytool.get_pinyin_id(token))
        _tokens.append("[SEP]")
        segment_ids.append(0)
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
            _lmask.append(0)

        assert len(input_ids) == self.max_sen_len
        assert len(input_mask) == self.max_sen_len
        assert len(segment_ids) == self.max_sen_len

        if ex_index < 5:
            tf.logging.info("*** Example ***")
            tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in _tokens]))
            tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            tf.logging.info("lmask: %s" % " ".join(map(str, _lmask)))

        feature = InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            lmask=_lmask
            )
        return feature
    
    def get_label_list(self):
        return self.label_list
 
    def file2features(self):
        output_file = self.tfrecord_path
        if os.path.exists(output_file):
            os.remove(output_file)
        examples = self.load_examples()
        n_examples = 0
        writer = tf.python_io.TFRecordWriter(output_file)
        for (ex_index, example) in enumerate(examples):
            if ex_index % 10000 == 0:
                tf.logging.info("Writing example %d" % ex_index)

            feature = self.convert_single_example(ex_index, example)
            create_int_feature = lambda values: tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            features = collections.OrderedDict()
            features["input_ids"] = create_int_feature(feature.input_ids)
            features["input_mask"] = create_int_feature(feature.input_mask)
            features["segment_ids"] = create_int_feature(feature.segment_ids)
            features["lmask"] = create_int_feature(feature.lmask)

            tf_example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(tf_example.SerializeToString())
            n_examples += 1
    
        self.num_examples = n_examples

    def build_data_generator(self, batch_size):
        def _decode_record(record):
            """Decodes a record to a TensorFlow example."""
            name_to_features = {
            "input_ids": tf.FixedLenFeature([self.max_sen_len], tf.int64),
            "input_mask": tf.FixedLenFeature([self.max_sen_len], tf.int64),
            "segment_ids": tf.FixedLenFeature([self.max_sen_len], tf.int64),
            "lmask": tf.FixedLenFeature([self.max_sen_len], tf.int64),
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
            lmask = example['lmask']
            
            masked_ids, masked_flgs, masked_py_ids, masked_sk_ids = tf.py_func(self.masker.mask_process, [input_ids], [tf.int32, tf.int32, tf.int32, tf.int32])
            lmask = tf.multiply(masked_flgs, lmask)
            label_ids = input_ids
            input_ids = masked_ids
            pinyin_ids = segment_ids
            masked_pinyin_ids = masked_py_ids

            return input_ids, input_mask, pinyin_ids, masked_pinyin_ids, masked_sk_ids, lmask, label_ids
        if self.is_training:
            dataset = tf.data.TFRecordDataset(filenames=self.TfrecordFile)
        else:
            pass
            #dataset = tf.data.TFRecordDataset(self.tfrecord_path)
        dataset = dataset.map(_decode_record, num_parallel_calls=10)
        if self.is_training:
            dataset = dataset.repeat().shuffle(buffer_size=500)
        dataset = dataset.batch(batch_size).prefetch(50)
        return dataset

if __name__ == '__main__':
    import sys
    vocab_file = sys.argv[1]
    text_file = sys.argv[2]
    output_file = sys.argv[3]
    dp = DataProcessor(input_path=text_file, max_sen_len=512, vocab_file=vocab_file, out_dir=output_file, is_training=False)
