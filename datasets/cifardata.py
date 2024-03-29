# derived from the tutorials in https://github.com/tensorflow/models
# &
# https://www.tensorflow.org/beta/guide/data_performance
#%%
import os
import shutil
import sys
import tarfile
import urllib.request

import tensorflow as tf
from six.moves import cPickle as pickle

from .utils import bytes_feature, int64_feature, write_as_tfrecords

HEIGHT = 32
WIDTH = 32
DEPTH = 3
CIFAR_NUM_CLASSES = 10

class Cifar:

    def __init__(self, CIFAR_URL=None):
        if CIFAR_URL is None:
            self.CIFAR_DOWNLOAD_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
        else:
            self.CIFAR_DOWNLOAD_URL = CIFAR_URL
        self.CIFAR_FILENAME = self.CIFAR_DOWNLOAD_URL.split('/')[-1]
        self.make_val_data = False

    def __download_data(self):
        if not os.path.isfile(self.CIFAR_FILENAME):
            urllib.request.urlretrieve(
                self.CIFAR_DOWNLOAD_URL, self.CIFAR_FILENAME)
        return self.CIFAR_FILENAME

    def _get_file_names(self):
        """Returns the file names expected to exist in the input_dir."""
        file_names = {}
        if self.make_val_data is False:
            file_names['train'] = ['data_batch_%d' % i for i in range(1, 6)]
        else :
            file_names['train'] = ['data_batch_%d' % i for i in range(1, 5)]
            file_names['validation'] = ['data_batch_5']
        file_names['eval'] = ['test_batch']
        return file_names

    def __read_pickle_from_file(self, filename):
        with open(filename, 'rb') as f:
            if sys.version_info >= (3, 0):
                data_dict = pickle.load(f, encoding='bytes')
            else:
                data_dict = pickle.load(f)
        return data_dict

    def __convert_to_tfrecord(self, input_files, output_file):
        """Converts a file to TFRecords."""
        print('Generating %s' % output_file)
        for input_file in input_files:
            data_dict = self.__read_pickle_from_file(input_file)
            data = data_dict[b'data']
            labels = data_dict[b'labels']
            write_as_tfrecords(data, labels, output_file)

    def __parse_fn(self, serialized_example):
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

    def generate_tfrecords(self, make_val_data=False):
        self.make_val_data = make_val_data
        CIFAR_LOCAL_FOLDER = 'cifar-10-batches-py'
        path = self.__download_data()
        tarfile.open(os.path.join(path), 'r:gz').extractall()
        file_names = self._get_file_names()
        input_dir = os.path.join('./', CIFAR_LOCAL_FOLDER)
        for mode, files in file_names.items():
            input_files = [os.path.join(input_dir, f) for f in files]
            output_file = os.path.join('./', mode + '.tfrecords')
            try:
                os.remove(output_file)
            except OSError:
                pass
            # Convert to tf.train.Example and write the to TFRecords.
            self.__convert_to_tfrecord(input_files, output_file)
        shutil.rmtree(CIFAR_LOCAL_FOLDER)
        print('Done!')

    def __make_tfr_dataset(self, FLAGS, tfrecordpath, repeat=True):
        files = tf.data.Dataset.list_files(tfrecordpath)
        dataset = files.interleave(
            tf.data.TFRecordDataset, cycle_length=FLAGS['num_parallel_reads'],
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # shuffle before repeat
        dataset = dataset.shuffle(buffer_size=FLAGS['shuffle_buffer_size'])
        if repeat:
            dataset = dataset.repeat()
        dataset = dataset.map(map_func=self.__parse_fn,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(batch_size=FLAGS['batch_size']).prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset

    def get_validation_dataset(self, FLAGS):
        if not self.make_val_data:
            raise Exception('validation data not created')
        dataset = self.__make_tfr_dataset(FLAGS, "./validation.tfrecords", repeat=False)
        return dataset

    def get_eval_dataset(self, FLAGS):
        dataset = self.__make_tfr_dataset(FLAGS, "./eval.tfrecords", repeat=False)
        return dataset

    def get_train_dataset(self, FLAGS):
        dataset = self.__make_tfr_dataset(FLAGS, "./train.tfrecords")
        return dataset

    def get_hwc(self):
        '''returns the image dimensions'''
        return HEIGHT, WIDTH, DEPTH
