#! /bin/bash

#convert text to tfrecord
vocab_file="./datas/vocab.txt"
train_corpus_file="./datas/pretrain_corpus_examples.txt"
output_tf_record_file="./train.tf_record"
python3 data_processor_mask.py $vocab_file $train_corpus_file $output_tf_record_file

#split tf_record_file to multiply files (for training on multiply gpus)
tf_records_dir=./train_tfrecords
mkdir $tf_records_dir
python3 split_records.py $output_tf_record_file $tf_records_dir
