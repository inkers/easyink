import tensorflow as tf
HEIGHT = 32
WIDTH = 32
DEPTH = 3
CIFAR_NUM_CLASSES = 10

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def write_as_tfrecords(data, labels, output_file):
    with tf.io.TFRecordWriter(output_file) as record_writer:
        num_entries_in_batch = len(labels)
        for i in range(num_entries_in_batch):
            example = tf.train.Example(features=tf.train.Features(
                feature={
                    'image': bytes_feature(data[i].tobytes()),
                    'label': int64_feature(labels[i])
                }))
            record_writer.write(example.SerializeToString())

def parse_fn(serialized_example):
    features = tf.io.parse_single_example(
        serialized_example,
        features={
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64),
        })
    image = tf.decode_raw(features['image'], tf.uint8)
    image.set_shape([DEPTH * HEIGHT * WIDTH])

    # Reshape from [num_channels * height * width] to [width, height, num_channels].
    image = tf.cast(
        tf.transpose(tf.reshape(image, [DEPTH, HEIGHT, WIDTH]), [1, 2, 0]),
        tf.float32)
    label = tf.cast(features['label'], tf.int32)
    label = tf.one_hot(label, CIFAR_NUM_CLASSES)
    return image, label

def load_tfr_dataset(tfrecordpath, FLAGS):
    files = tf.data.Dataset.list_files(tfrecordpath)
    dataset = files.interleave(
        tf.data.TFRecordDataset, cycle_length=FLAGS['num_parallel_reads'],
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # shuffle before repeat
    dataset = dataset.shuffle(
        buffer_size=FLAGS['shuffle_buffer_size']).repeat()
    dataset = dataset.map(map_func=parse_fn,
                            num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size=FLAGS['batch_size']).prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

def load_dataset_np(X, Y, FLAGS):
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.shuffle(
        buffer_size=FLAGS['shuffle_buffer_size']).repeat()
    dataset = dataset.map(lambda x, y: (tf.div(tf.cast(x, tf.float32), 255.0), tf.reshape(tf.one_hot(y, 10), (-1, 10))))
    dataset = dataset.map(lambda x, y: (tf.image.random_flip_left_right(x), y))
    
    dataset = dataset.batch(batch_size=FLAGS['batch_size']).prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset
