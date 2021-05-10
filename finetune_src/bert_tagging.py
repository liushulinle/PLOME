#-*-coding:utf8-*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
import random
from collections import namedtuple
import re
import numpy as np
import modeling
import optimization
import tokenization
import tensorflow as tf
from data_processor_tagging import DataProcessor

class BertTagging:
    def __init__(self, bert_config_path, num_class, pyid2seq, skid2seq, py_dim, max_sen_len, py_or_sk, keep_prob=0.9, multi_task=True, zi_py_matrix=None):
        self.num_class = num_class
        self.keep_prob = keep_prob
        self.pyid2seq = pyid2seq
        self.skid2seq = skid2seq
        self.PYLEN = 4
        self.SKLEN = 10
        self.MAX_SEN_LEN = max_sen_len
        self.PYDIM = py_dim 
        self.py_or_sk = py_or_sk  
        self.zi_py_matrix = zi_py_matrix
        self.multi_task = multi_task
        self.bert_config = modeling.BertConfig.from_json_file(bert_config_path)
        
    def create_model(self, input_ids, input_mask, segment_ids, stroke_ids, lmask, labels, py_labels, is_training):
        def lstm_op(sen_pyids, ZM_EMBS, ID2SEQ, flg):
            hidden_size = 768
            seq_len = self.PYLEN if 'py' in flg else self.SKLEN
            sen_pyids = tf.reshape(sen_pyids, shape=[-1])
            sen_seq = tf.nn.embedding_lookup(ID2SEQ, sen_pyids, name="lookup_pyid2seq")
            sen_seq_emb = tf.nn.embedding_lookup(ZM_EMBS, sen_seq, name="lookup_pyemb")
            sen_seq_emb = tf.reshape(sen_seq_emb, shape=[-1, seq_len, self.PYDIM]) 

            with tf.variable_scope('GRU', reuse=tf.AUTO_REUSE):
                cell = tf.contrib.rnn.GRUCell(num_units=hidden_size)
                all_out, final_out = tf.nn.dynamic_rnn(cell, sen_seq_emb, dtype=tf.float32)
            lstm_output = tf.reshape(final_out, shape=[-1, self.MAX_SEN_LEN, hidden_size])
            return lstm_output

        pinyin_ids = segment_ids
        if 'py' in self.py_or_sk:                
            with tf.variable_scope('py_emb', reuse=tf.AUTO_REUSE):
                zimu_emb = tf.get_variable('zimu_emb', [30, self.PYDIM], initializer=tf.truncated_normal_initializer(stddev=0.02))
                id2seq = tf.get_variable("pyid2seq_matrix", initializer=self.pyid2seq, dtype=tf.int32, trainable=False)
            py_embs = lstm_op(pinyin_ids, zimu_emb, id2seq, 'py')
        elif 'sk' in self.py_or_sk:
            with tf.variable_scope('py_emb', reuse=tf.AUTO_REUSE):
                zimu_emb = tf.get_variable('zimu_emb', [30, self.PYDIM], initializer=tf.truncated_normal_initializer(stddev=0.02))
                id2seq = tf.get_variable("pyid2seq_matrix", initializer=self.skid2seq, dtype=tf.int32, trainable=False)
            py_embs = lstm_op(stroke_ids, zimu_emb, id2seq, 'sk')
        elif 'all' in self.py_or_sk:
            with tf.variable_scope('py_emb', reuse=tf.AUTO_REUSE):
                zimu_emb = tf.get_variable('zimu_emb', [30, self.PYDIM], initializer=tf.truncated_normal_initializer(stddev=0.02))
                pyid2seq = tf.get_variable("pyid2seq_matrix", initializer=self.pyid2seq, dtype=tf.int32, trainable=False)
                py_embs = lstm_op(pinyin_ids, zimu_emb, pyid2seq, 'py')
            with tf.variable_scope('sk_emb', reuse=tf.AUTO_REUSE):
                zisk_emb = tf.get_variable('zisk_emb', [7, self.PYDIM], initializer=tf.truncated_normal_initializer(stddev=0.02))
                skid2seq = tf.get_variable("pyid2seq_matrix", initializer=self.skid2seq, dtype=tf.int32, trainable=False)
                sk_embs = lstm_op(stroke_ids, zisk_emb, skid2seq, 'sk')
            py_embs = py_embs + sk_embs
        else:
            raise Exception('not supported py_or_sk:%s' % self.py_or_sk)
 


        if lmask is None: #for predict, not use these placeholders
            lmask = input_ids
            labels = input_ids
            py_labels = input_ids
        
        with tf.variable_scope('bert', reuse=tf.AUTO_REUSE):
            model = modeling.BertModel(
            config=self.bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            pinyin_embs=py_embs,
            use_one_hot_embeddings=False)

            output_seq = model.get_all_encoder_layers()[-1]
            hidden_size = output_seq[-1].shape[-1].value

        if (self.multi_task is True):
            with tf.variable_scope('zi_py_matrix', reuse=tf.AUTO_REUSE):
                zi_py_mapping = tf.get_variable("zi_py_matrix", initializer=self.zi_py_matrix.T, trainable=False)

        with tf.variable_scope("loss", reuse=tf.AUTO_REUSE):
            output_weights = tf.get_variable(
            "output_weights", [self.num_class, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

            output_bias = tf.get_variable(
            "output_bias", [self.num_class], initializer=tf.zeros_initializer())

            if self.multi_task is True:
                output_py_weights = tf.get_variable(
                "output_py_weights", [430, hidden_size],
                initializer=tf.truncated_normal_initializer(stddev=0.02))

                output_py_bias = tf.get_variable(
                "output_py_bias", [430], initializer=tf.zeros_initializer()) 

            output = tf.reshape(output_seq, [-1, hidden_size])
            labels = tf.squeeze(tf.reshape(labels, [-1, 1]))
            mask = tf.squeeze(tf.reshape(lmask, [-1, 1]))
            if is_training:
                output = tf.nn.dropout(output, keep_prob=self.keep_prob)
            #  loss
            logits = tf.matmul(output, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            probabilities = tf.nn.softmax(logits, axis=-1)
            log_probs = tf.nn.log_softmax(logits, axis=-1)

            one_hot_labels = tf.one_hot(labels, depth=self.num_class, dtype=tf.float32)
            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1) * tf.cast(mask, tf.float32)
            loss = tf.reduce_sum(per_example_loss) / tf.reduce_sum(tf.cast(mask, tf.float32))
            
            # py loss
            py_probs = tf.zeros(shape=[1,1])
            py_one_hot_labels = tf.zeros(shape=[1,1])
            fusion_prob = tf.zeros(shape=[1,1])
            if self.multi_task is True:
                py_labels = tf.squeeze(tf.reshape(py_labels, [-1, 1]))
                py_logits = tf.matmul(output, output_py_weights, transpose_b=True)
                py_logits = tf.nn.bias_add(py_logits, output_py_bias)
                py_probs = tf.nn.softmax(py_logits, axis=-1)  
                py_log_probs = tf.nn.log_softmax(py_logits, axis=-1)
                py_one_hot_labels = tf.one_hot(py_labels, depth=430, dtype=tf.float32)
                py_per_example_loss = -tf.reduce_sum(py_one_hot_labels * py_log_probs, axis=-1) * tf.cast(mask, tf.float32)
                py_loss = tf.reduce_sum(py_per_example_loss) / tf.reduce_sum(tf.cast(mask, tf.float32))
                loss = loss + py_loss
                
                if is_training is False:
                    fusion_prob = tf.matmul(py_probs, zi_py_mapping)
                    fusion_prob = fusion_prob * probabilities
     

            return (loss, probabilities, one_hot_labels, mask, py_probs, py_one_hot_labels, fusion_prob)

