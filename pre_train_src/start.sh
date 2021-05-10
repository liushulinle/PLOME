#!/bin/sh
appname=pretrain_plome
py_dim=32
multi_task=0
gpuid_list="0,1,2,3,4,5"
export logdir='./'
train_tfrecord_path="./train_tfrecords"
output_dir="./${appname}_output"
keep_prob=0.9
max_sen_len=512
init_checkpoint="./google_bert" #"0 for random initialization"
vocab_file="./datas/vocab.txt"
bert_config_file="./datas/bert_config.json"
batch_size=8
epoch=10
learning_rate=5e-5 

mkdir $output_dir
mkdir $train_tfrecord_path

horovodrun -np 6 -H localhost:6 python3 ./train_masklm.py --vocab_file $vocab_file --bert_config_path $bert_config_file --gpuid_list $gpuid_list --train_path $train_path --output_dir $output_dir --max_sen_len $max_sen_len --batch_size $batch_size --learning_rate $learning_rate --epoch $epoch --keep_prob $keep_prob --py_dim $py_dim --train_tfrecord_path $train_tfrecord_path --init_checkpoint $init_checkpoint --multi_task $multi_task


