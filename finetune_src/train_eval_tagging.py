#-*-coding:utf8-*-

import sys, os
import numpy as np
import tensorflow as tf
from bert_tagging import DataProcessor, BertTagging
import modeling
import optimization
import time
from tagging_eval import score_f, score_f_sent, score_f_py
tf.logging.set_verbosity(tf.logging.ERROR)

def evaluate(FLAGS, sess, model, data_processor, label_list=None):
    gpuid = FLAGS.gpuid
    max_sen_len = FLAGS.max_sen_len
    train_path = FLAGS.train_path
    test_file = FLAGS.test_path
    out_dir = FLAGS.output_dir
    batch_size = 50
    EPOCH = FLAGS.epoch
    learning_rate = FLAGS.learning_rate
    init_bert_dir = FLAGS.init_bert_path
    learning_rate = FLAGS.learning_rate
    vocab_file = '%s/vocab.txt' % init_bert_dir
    init_checkpoint = '%s/bert_model.ckpt' % init_bert_dir
    bert_config_path = '%s/bert_config.json'% init_bert_dir
 

    test_num = data_processor.num_examples
    test_data = data_processor.build_data_generator(batch_size)
    iterator = test_data.make_one_shot_iterator()
    input_ids, input_mask, segment_ids, stroke_ids, lmask, label_ids, py_labels = iterator.get_next()

    (pred_loss, pred_probs, gold_probs, gold_mask, py_probs, py_one_hot_labels, fusion_prob) = \
               model.create_model(input_ids, input_mask, segment_ids, stroke_ids, lmask, label_ids, py_labels, is_training=False)
    label_list = data_processor.label_list
    py_label_list = data_processor.py_label_list
    ans_c, ans_py, ans = [], [], []
    all_inputs, all_golds, all_preds = [], [], []
    all_py_golds, all_py_preds = [], []
    all_fusino_preds = []
    all_inputs_sent, all_golds_sent, all_preds_sent = [], [], []
    all_py_pred_sent, all_py_gold_sent, all_fusion_sent = [], [], []
    all_py_inputs, all_py_inputs_sent = [], []
    for step in range(test_num // batch_size):
        if model.multi_task is True: 
            inputs, py_inputs, loss_value, preds, golds, gmask, py_pred, py_golds, fusion_pred = sess.run([input_ids, segment_ids, pred_loss, pred_probs, gold_probs, gold_mask, py_probs, py_one_hot_labels, fusion_prob])
        else:
            inputs, loss_value, preds, golds, gmask = sess.run([input_ids, pred_loss, pred_probs, gold_probs, gold_mask])

        preds = np.reshape(preds, (batch_size, max_sen_len, len(label_list))) 
        preds = np.argmax(preds, axis=2)
        golds = np.reshape(golds, (batch_size, max_sen_len, len(label_list))) 
        golds = np.argmax(golds, axis=2)
        gmask = np.reshape(gmask, (batch_size, max_sen_len))
        if model.multi_task is True:
            py_pred = np.reshape(py_pred, (batch_size, max_sen_len, 430))
            py_pred = np.argmax(py_pred, axis=2)
            py_golds = np.reshape(py_golds, (batch_size, max_sen_len, 430))
            py_golds = np.argmax(py_golds, axis=2)
            fusion_pred = np.reshape(fusion_pred, (batch_size, max_sen_len, len(label_list)))
            fusion_pred = np.argmax(fusion_pred, axis=2)
        for k in range(batch_size):
            tmp1, tmp2, tmp3, tmps4, tmps5, tmps6, tmps7 = [], [], [], [], [], [], []
            for j in range(max_sen_len):
                if gmask[k][j] == 0: continue
                all_golds.append(golds[k][j])
                all_preds.append(preds[k][j])
                all_inputs.append(inputs[k][j])
                tmp1.append(label_list[golds[k][j]])
                tmp2.append(label_list[preds[k][j]])
                tmp3.append(label_list[inputs[k][j]])
                if model.multi_task is True:
                    all_py_inputs.append(py_inputs[k][j])
                    all_py_golds.append(py_golds[k][j])
                    all_py_preds.append(py_pred[k][j])
                    all_fusino_preds.append(fusion_pred[k][j])
                    tmps4.append(str(py_golds[k][j]))
                    tmps5.append(str(py_pred[k][j]))
                    tmps6.append(label_list[fusion_pred[k][j]])
                    tmps7.append(str(py_inputs[k][j]))
                    
                
            all_golds_sent.append(tmp1)
            all_preds_sent.append(tmp2)
            all_inputs_sent.append(tmp3)
            if model.multi_task is True:
                all_py_pred_sent.append(tmps4)
                all_py_gold_sent.append(tmps5)
                all_fusion_sent.append(tmps6)
                all_py_inputs_sent.append(tmps7)
                

    all_golds = [label_list[k] for k in all_golds]
    all_preds = [label_list[k] for k in all_preds]
    all_inputs = [label_list[k] for k in all_inputs]
    if model.multi_task is True:
        all_fusino_preds = [label_list[k] for k in all_fusino_preds]
        all_py_inputs = [py_label_list.get(int(k), k) for k in all_py_inputs]
        all_py_golds = [py_label_list.get(int(k), k) for k in all_py_golds]
        all_py_preds = [py_label_list.get(int(k), k) for k in all_py_preds]
   
    if model.multi_task is True:
        print('pinyin result:')
        score_f_py((all_py_inputs, all_py_golds, all_py_preds), (all_inputs, all_golds, all_preds), out_dir, False)
        print('fusion result:')
        p, r, f = score_f((all_inputs, all_golds, all_fusino_preds))
        score_f_sent(all_inputs_sent, all_golds_sent, all_fusion_sent)
    else:    
        print('zi result:') 
        p, r, f = score_f((all_inputs, all_golds, all_preds), only_check=False)
        p_sent, r_sent, f_sent = score_f_sent(all_inputs_sent, all_golds_sent, all_preds_sent)
 
    del data_processor
    return f


def train(FLAGS):
    rand_type_emb = False

    gpuid = FLAGS.gpuid
    max_sen_len = FLAGS.max_sen_len
    train_path = FLAGS.train_path
    test_file = FLAGS.test_path
    out_dir = FLAGS.output_dir
    batch_size = FLAGS.batch_size
    EPOCH = FLAGS.epoch
    init_bert_dir = FLAGS.init_bert_path
    learning_rate = FLAGS.learning_rate
    sk_or_py = FLAGS.sk_or_py
    multi_task = True if FLAGS.multi_task > 0 else False
    py_dim = int(FLAGS.py_dim)
    vocab_file = '%s/vocab.txt' % init_bert_dir
    init_checkpoint = '%s/bert_model.ckpt' % init_bert_dir
    bert_config_path = '%s/bert_config.json'% init_bert_dir
 
    if os.path.exists(out_dir) is False:
        os.mkdir(out_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = gpuid
    keep_prob = FLAGS.keep_prob
    print('test_file=', test_file)
    test_data_processor = DataProcessor(test_file, max_sen_len, vocab_file, out_dir, label_list=None, is_training=False)
    print('train_file=', train_path)
    data_processor = DataProcessor(train_path, max_sen_len, vocab_file, out_dir, label_list=None, is_training=True)

    zi_py_matrix = data_processor.get_zi_py_matrix()
    train_num = data_processor.num_examples
    train_data = data_processor.build_data_generator(batch_size)
    iterator = train_data.make_one_shot_iterator()
    input_ids, input_mask, segment_ids, stroke_ids, lmask, label_ids, py_labels = iterator.get_next()

    model = BertTagging(bert_config_path, num_class=len(data_processor.get_label_list()), pyid2seq=data_processor.PYID2SEQ, skid2seq=data_processor.SKID2SEQ, py_dim=py_dim, max_sen_len=max_sen_len, py_or_sk=sk_or_py,  keep_prob=keep_prob, zi_py_matrix=zi_py_matrix, multi_task=multi_task)
    (loss, probs, golds, _, py_probs, py_one_hot_labels, fusion_prob) = model.create_model(input_ids, input_mask, segment_ids, stroke_ids, lmask, label_ids, py_labels, is_training=True)

    tf_config = tf.ConfigProto(log_device_placement=False)
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        if init_checkpoint is not None:
            tvars = tf.trainable_variables()
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            keys = [x for x in assignment_map.keys()]
            for key in keys:
                if (rand_type_emb is True) and ('token_type_embeddings' in key):
                    del assignment_map[key]
                    continue
                print(key, '\t', assignment_map[key])
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)


        num_steps = train_num // batch_size * EPOCH
        num_warmup_steps = num_steps // 10
        train_op = optimization.create_optimizer(loss, learning_rate, num_steps, num_warmup_steps, use_tpu=False)

        init = tf.global_variables_initializer()
        sess.run(init)
 
        loss_values = []
        saver = tf.train.Saver()
        best_score = 0.0
        best_model_path = os.path.join(out_dir, 'best.ckpt')
        total_step = 0
        for epoch in range(EPOCH):
            for step in range(int(train_num / batch_size)):
                total_step += 1
                start_time = time.time()
                train_loss, _ = sess.run([loss,  train_op]) 
                loss_values.append(train_loss)
                if step % 500 == 0:
                    duration = time.time() - start_time
                    examples_per_sec = float(duration) / batch_size
                    format_str = ('Epoch {} step {},  train loss = {:.4f},{:.4f},{:.4f} ( {:.4f} examples/sec; {:.4f} ''sec/batch)')
                    print (format_str.format(epoch, step, np.mean(loss_values),np.mean(loss_values[-1000:]),np.mean(loss_values[-100:]), examples_per_sec, duration))
                    loss_values = loss_values[-1000:]

                    f1 = evaluate(FLAGS, sess, model, test_data_processor)
                    if f1 > best_score:
                        saver.save(sess, best_model_path)
                        best_score = f1
                    sys.stdout.flush()
            f1 = evaluate(FLAGS, sess, model, test_data_processor)
            if f1 > best_score:
                saver.save(sess, best_model_path)
                best_score = f1
            sys.stdout.flush()
        print ('best f value:', best_score)
 
 
if __name__ == '__main__':

    flags = tf.flags
    ## Required parameters
    flags.DEFINE_string("gpuid", '0', "The gpu NO. ")

    ## Optional
    flags.DEFINE_string("train_path", '', "train path ")
    flags.DEFINE_string("test_path", '', "test path ")
    flags.DEFINE_string("output_dir", '', "out dir ")
    flags.DEFINE_string("init_bert_path", '', "out dir ")
    flags.DEFINE_string("sk_or_py", 'py', "sk_or_py")
    flags.DEFINE_string("label_list", '', 'max_sen_len')
    flags.DEFINE_integer("max_sen_len", 64, 'max_sen_len')
    flags.DEFINE_integer("batch_size", 32, 'batch_size')
    flags.DEFINE_integer("single_text", '0', 'single_text')
    flags.DEFINE_integer("epoch", 2, 'batch_size')
    flags.DEFINE_float("learning_rate", 5e-5, 'filter_punc')
    flags.DEFINE_float("keep_prob", 0.9, 'keep prob in dropout')
    flags.DEFINE_integer("py_dim", 32, 'keep prob in dropout')
    flags.DEFINE_integer("multi_task", 0, 'keep prob in dropout')


    flags.mark_flag_as_required('gpuid')
    flags.mark_flag_as_required('train_path')
    flags.mark_flag_as_required('test_path')
    flags.mark_flag_as_required('init_bert_path')
    flags.mark_flag_as_required('output_dir')
    flags.mark_flag_as_required('label_list')
    flags.mark_flag_as_required('max_sen_len')

    FLAGS = flags.FLAGS
    print ('Confings:')
    print ('\tlearning_rate=', FLAGS.learning_rate)
    print ('\ttrain_path=', FLAGS.train_path)
    print ('\ttest_path=', FLAGS.test_path)
    print ('\tinit_bert_path=', FLAGS.init_bert_path)
    print ('\toutput_dir=', FLAGS.output_dir)
    print ('\tmax_sen_len=', FLAGS.max_sen_len)
    print ('\tpy_dim=', FLAGS.py_dim)
    print ('\tmulti_task=', FLAGS.multi_task)
    print ('\tsk_or_py=', FLAGS.sk_or_py)
    train(FLAGS)

