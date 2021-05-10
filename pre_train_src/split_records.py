import tensorflow as tf

def split_tfrecord(tfrecord_path, out_dir, n_each_file_samples):
    with tf.Graph().as_default(), tf.Session() as sess:
        ds = tf.data.TFRecordDataset(tfrecord_path).batch(1000)
        batch = ds.make_one_shot_iterator().get_next()
        part_num = 0
        n_writed = 0
        part_path = out_dir + '/{:02d}.tfrecord'.format(part_num)
        writer = tf.python_io.TFRecordWriter(part_path)
        while True:
            try:
                records = sess.run(batch)
                for record in records:
                    writer.write(record)
                    n_writed += 1
                if n_writed > n_each_file_samples:        
                    part_num += 1
                    n_writed = 0
                    writer.close()
                    part_path = out_dir + '/{:02d}.tfrecord'.format(part_num)
                    writer = tf.python_io.TFRecordWriter(part_path)
            except tf.errors.OutOfRangeError: break
        writer.close()



if __name__ == '__main__':
    import sys
    in_rec = sys.argv[1]
    out_dir = sys.argv[2]
    split_tfrecord(in_rec, out_dir, 5000000)
