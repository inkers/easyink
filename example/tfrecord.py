#%%
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

import datasets.cifardata as cfdata

FLAGS = {
    'shuffle_buffer_size': 100,
    'num_parallel_reads': 100,
    'batch_size': 512
}
cifar = cfdata.Cifar()
cifar.generate_tfrecords()
train_dataset = cifar.get_train_dataset(FLAGS)

for train_images_batch, train_labels_batch in train_dataset:
  # check if able to loop
  break
