#%%
import datasets.cifardata as cfdata


FLAGS = {
    'shuffle_buffer_size': 100,
    'num_parallel_reads': 100,
    'batch_size': 512
}
cifar = cfdata.Cifar()
cifar.generate_tfrecords()
dataset = cifar.get_train_dataset(FLAGS)
