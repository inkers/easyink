import os
import sys
import xrange

from six.moves import cPickle as pickle
import tarfile
import tensorflow as tf
import urllib.request

CIFAR_FILENAME = 'cifar-10-python.tar.gz'
CIFAR_LOCAL_FOLDER = 'cifar-10-batches-py'

def download_cifar_data():
  CIFAR_DOWNLOAD_URL = 'https://www.cs.toronto.edu/~kriz/' + CIFAR_FILENAME
  if not os.path.isfile(CIFAR_FILENAME):
    urllib.request.urlretrieve (CIFAR_DOWNLOAD_URL, CIFAR_FILENAME)
  return CIFAR_FILENAME

def _get_file_names():
  """Returns the file names expected to exist in the input_dir."""
  file_names = {}
  file_names['train'] = ['data_batch_%d' % i for i in xrange(1, 5)]
  file_names['validation'] = ['data_batch_5']
  file_names['eval'] = ['test_batch']
  return file_names

def read_pickle_from_file(filename):
  with open(filename, 'rb') as f:
    if sys.version_info >= (3, 0):
      data_dict = pickle.load(f, encoding='bytes')
    else:
      data_dict = pickle.load(f)
  return data_dict

def convert_to_tfrecord(input_files, output_file):
  """Converts a file to TFRecords."""
  print('Generating %s' % output_file)
  with tf.python_io.TFRecordWriter(output_file) as record_writer:
    for input_file in input_files:
      data_dict = read_pickle_from_file(input_file)
      data = data_dict[b'data']
      labels = data_dict[b'labels']
      num_entries_in_batch = len(labels)
      for i in range(num_entries_in_batch):
        example = tf.train.Example(features=tf.train.Features(
            feature={
                'image': _bytes_feature(data[i].tobytes()),
                'label': _int64_feature(labels[i])
            }))
        record_writer.write(example.SerializeToString())


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

path , _ = download_cifar_data()
tarfile.open(os.path.join(path), 'r:gz').extractall()
file_names = _get_file_names()
input_dir = os.path.join('./', CIFAR_LOCAL_FOLDER)
for mode, files in file_names.items():
  input_files = [os.path.join(input_dir, f) for f in files]
  output_file = os.path.join('./', mode + '.tfrecords')
  try:
    os.remove(output_file)
  except OSError:
    pass
  # Convert to tf.train.Example and write the to TFRecords.
  convert_to_tfrecord(input_files, output_file)
print('Done!')