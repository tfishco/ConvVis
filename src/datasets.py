from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet
from tensorflow.python.framework import dtypes

class MNIST_Data:
    def __init__(self):
        mnist = input_data.read_data_sets('/src/resource/MNIST_data', one_hot=True)

        self.train_dataset = mnist.train

        self.test_dataset = mnist.test

        self.image_dimensions = 28

class CIFAR_Data:
    def __init__(self):
        import cifar_10
        cifar_10.load_and_preprocess_input(dataset_dir='resource/CIFAR_data')

        self.test_dataset = DataSet(cifar_10.validate_all['data'],
              cifar_10.validate_all['labels'],reshape=False)

        self.train_dataset = DataSet(cifar_10.train_all['data'],
        cifar_10.train_all['labels'], reshape=False)

        self.image_dimensions = cifar_10.image_width
