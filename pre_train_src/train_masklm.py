#-*-coding:utf8-*-

import sys, os
import numpy as np
import tensorflow as tf
from mask_lm import DataProcessor, MaskLM
import modeling
import optimization
import time
import random
#tf.logging.set_verbosity(tf.logging.INFO)
tf.logging.set_verbosity(tf.logging.ERROR)

class MConfig:
    pass

def train(FLAGS):
    PY_OR_SK = 'all'

    rand_type_emb = True
    args = MConfig()
    args.use_mgpu = False
    args.seed = 1
    args.py_dim = int(FLAGS.py_dim)
    args.multi_task = True if int(FLAGS.multi_task) > 0 else False
    if int(FLAGS.use_mgpu) > 0:
        args.use_mgpu = True
        import horovod.tensorflow as hvd
        hvd.init()
        args.seed = hvd.rank()
        args.hvd = hvd
        print("=========== rank: ", hvd.rank(), ", local rank: ", hvd.local_rank(), ", size: ", hvd.size(), ", seed: ", args.seed)
    init_checkpoint = None if len(FLAGS.init_checkpoint.strip()) < 3 else FLAGS.init_checkpoint.strip()
    tf.random.set_random_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
     
    gpuid_list = FLAGS.gpuid_list.strip().split(',')

    max_sen_len = FLAGS.max_sen_len
    train_path = FLAGS.train_path
    test_file = FLAGS.test_path
    out_dir = FLAGS.output_dir
    train_tfrecord_dir = FLAGS.train_tfrecord_path
    batch_size = FLAGS.batch_size
    bert_config_path = FLAGS.bert_config_path
    EPOCH = FLAGS.epoch
    learning_rate = FLAGS.learning_rate
    vocab_file = FLAGS.vocab_file

    keep_prob = FLAGS.keep_prob
    data_processor = DataProcessor(train_path, max_sen_len, vocab_file, train_tfrecord_dir, label_list=None, is_training=True)
    train_num = data_processor.num_examples
    train_data = data_processor.build_data_generator(batch_size)
    iterator = train_data.make_one_shot_iterator()
    input_ids, input_mask, pinyin_ids, masked_pinyin_ids, masked_stroke_ids, lmask, label_ids = iterator.get_next()

    #print ('input-ids:', id(input_ids), input_ids)

    input_ids.set_shape([None, max_sen_len])
    input_mask.set_shape([None, max_sen_len])
    pinyin_ids.set_shape([None, max_sen_len])
    lmask.set_shape([None, max_sen_len])
    label_ids.set_shape([None, max_sen_len])
    masked_pinyin_ids.set_shape([None, max_sen_len])


        
    tf_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
    if args.use_mgpu is True:
        tf_config.gpu_options.visible_device_list = str(hvd.local_rank())

    model = MaskLM(bert_config_path, num_class=len(data_processor.get_label_list()), pyid2seq=data_processor.PYID2SEQ, skid2seq=data_processor.SKID2SEQ, py_dim=args.py_dim, py_or_sk=PY_OR_SK, keep_prob=keep_prob, multi_task=args.multi_task)
    (loss, probs, golds, _, py_loss) = model.create_model(input_ids, input_mask, 
                                              masked_pinyin_ids, masked_stroke_ids, lmask, label_ids, pinyin_ids, is_training=True)
 

    num_steps = train_num // batch_size * EPOCH
    num_warmup_steps = 100000
    #if args.use_mgpu is True:
    #    learning_rate = learning_rate * hvd.size()
    train_op = optimization.create_optimizer(loss, learning_rate, num_steps, num_warmup_steps, args)


    with tf.Session(config=tf_config) as sess:
        if init_checkpoint is not None:
            print ('google_bert_init')
            tvars = tf.trainable_variables()
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint + '/bert_model.ckpt')
            keys = [x for x in assignment_map]
            for k in keys:
                if (rand_type_emb is True) and ('token_type_embeddings' in k):
                    del assignment_map[k]
                    continue
                print(k, '\t', assignment_map[k])

            tf.train.init_from_checkpoint(init_checkpoint + '/bert_model.ckpt', assignment_map)
        init = tf.global_variables_initializer()
        sess.run(init)
        
        if args.use_mgpu is True:
            sess.run(hvd.broadcast_global_variables(0))
 
        loss_values = []
        saver = tf.train.Saver()
        best_score = 0.0
        best_model_path = os.path.join(out_dir, 'bert_model.ckpt')
        total_step = 0
        for epoch in range(EPOCH):
            for step in range(int(train_num / batch_size)):
                total_step += 1
                start_time = time.time()
                train_loss, _ = sess.run([loss,  train_op]) 
                loss_values.append(train_loss)
                if step % 50 == 0:
                    duration = time.time() - start_time
                    examples_per_sec = float(duration) / batch_size
                    format_str = ('Epoch {} step {},  train loss = {:.4f},{:.4f},{:.4f} ( {:.4f} examples/sec; {:.4f} ''sec/batch)')
                    
                    if hvd.rank() == 0:
                        print (format_str.format(epoch, step, np.mean(loss_values),np.mean(loss_values[-1000:]),np.mean(loss_values[-100:]), examples_per_sec, duration))
                    loss_values = loss_values[-1000:]
                if step % 1000 == 0 and hvd.rank() == 0:
                    saver.save(sess, best_model_path)
 
 
if __name__ == '__main__':

    flags = tf.flags
    ## Required parameters
    flags.DEFINE_string("gpuid_list", '0', "i.e:0,1,2")
    PREFIX = './pretrain_data'
    ## Optional
    flags.DEFINE_string("train_path", '', "train path ")
    flags.DEFINE_string("test_path", '', "test path ")
    flags.DEFINE_string("train_tfrecord_path", '', "train path ")
    flags.DEFINE_string("output_dir", '', "out dir ")
    flags.DEFINE_string("vocab_file", '%s/datas/bert_datas/vocab.txt' % PREFIX, 'vocab')
    flags.DEFINE_string("init_checkpoint", '', '')
    flags.DEFINE_string("bert_config_path", '%s/datas/bert_datas/bert_config.json' % PREFIX, '')
    flags.DEFINE_string("label_list", '', 'max_sen_len')
    flags.DEFINE_integer("max_sen_len", 64, 'max_sen_len')
    flags.DEFINE_integer("batch_size", 32, 'batch_size')
    flags.DEFINE_integer("py_dim", 1, 'use_pinyin')
    flags.DEFINE_integer("multi_task", 1, 'multi_task')
    flags.DEFINE_integer("epoch", 2, 'batch_size')
    flags.DEFINE_float("learning_rate", 5e-5, 'filter_punc')
    flags.DEFINE_float("keep_prob", 0.9, 'keep prob in dropout')
    flags.DEFINE_string("use_mgpu", '1', 'keep prob in dropout')


    flags.mark_flag_as_required('gpuid_list')
    flags.mark_flag_as_required('train_path')
    flags.mark_flag_as_required('output_dir')
    flags.mark_flag_as_required('train_tfrecord_path')
    flags.mark_flag_as_required('max_sen_len')

    FLAGS = flags.FLAGS
    #FLAGS.bert_config_path = '%s/bert_config.json' % FLAGS.output_dir
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpuid_list
    print ('Confings:')
    print ('\tgpuid_list=', FLAGS.gpuid_list)
    print ('\ttrain_path=', FLAGS.train_path)
    print ('\ttest_path=', FLAGS.test_path)
    print ('\toutput_dir=', FLAGS.output_dir)
    print ('\tmax_sen_len=', FLAGS.max_sen_len)
    print ('\tbert_config_path=', FLAGS.bert_config_path)
    print ('\tmulti_task=', FLAGS.multi_task)
    train(FLAGS)

