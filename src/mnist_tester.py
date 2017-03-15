from __future__ import print_function
import tensorflow as tf
from flask import Flask, render_template, request, send_file
from tensorflow.examples.tutorials.mnist import input_data
import random
import numpy as np
import json
import network_json
import sys
from matplotlib import pyplot as plt

sys.path.insert(0, 'pre-trained/')

import classifier

mnist = input_data.read_data_sets('resource/MNIST_data', one_hot=True)

x = tf.placeholder("float", [784])

sess = tf.Session()

with tf.variable_scope("conv"):
    prediction , variables, features, separate_conv = classifier.conv(x, 1.0)
saver = tf.train.Saver(variables)
saver.restore(sess, "pre-trained/mnist/graph_mnist100/mnist.ckpt")

index = 0

image = mnist.test.images[index]
label = mnist.test.labels[index]
#(1, 32, 1, 28, 28, 1)
#(32, 64, 1, 14, 14, 1)
conv_array = np.array(sess.run(separate_conv, feed_dict={x:image})[1])
print(conv_array.shape)
conv_array = conv_array.transpose((2,5,0,1,3,4)).squeeze()
print(conv_array.shape)
zeroth_index = conv_array[0][0]
plt.imshow(zeroth_index, interpolation='nearest',cmap='gray')
plt.show()
