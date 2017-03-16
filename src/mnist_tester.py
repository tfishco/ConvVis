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
import cifar_10

mnist = input_data.read_data_sets('resource/MNIST_data', one_hot=True)

cifar_10.load_and_preprocess_input(dataset_dir='src/resource/CIFAR_data')

x = tf.placeholder("float", [784])

sess = tf.Session()

with tf.variable_scope("conv"):
    prediction , variables, features, separate_conv = classifier.conv(x, 1.0)
saver = tf.train.Saver(variables)
saver.restore(sess, "pre-trained/mnist/graph_mnist100/mnist.ckpt")

def get_separate_conv_data(data, threshold):
    separate = {}
    #np.round(np.multiply(get_feature_map(feature_list[1], 28, 32), 255), decimals=0)
    sep_1 = np.array([np.array(data[0]).transpose((2,5,0,1,3,4)).squeeze().tolist()])
    sep_2 = np.array(data[1]).transpose((2,5,0,1,3,4)).squeeze()
    separate['separate_conv1'] = get_highest_layer_activations(threshold,np.multiply(sep_1,255))
    separate['separate_conv2'] = get_highest_layer_activations(threshold,np.multiply(sep_2,255))#np.array(data[1]).transpose((2,5,0,1,3,4)).squeeze().shape()
    return separate

def get_highest_layer_activations(threshold, data):
    brightness = []
    for i in range(len(data)):
        group_brightness = []
        for j in range(len(data[i])):
            val = abs(get_image_brightness(np.array(data[i][j]).flatten()))
            group_brightness.append(val)
        brightness.append(group_brightness)
    top_brightness = []
    for i in range(len(brightness)):
        max_indexes = np.array(brightness[i]).argsort()[-threshold:][::-1]
        max_vals = []
        for j in range(threshold):
            max_vals.append([brightness[i][max_indexes[j]],max_indexes[j]])
        top_brightness.append(max_vals)
    return top_brightness

def get_image_brightness(image):
    total_brightness = 0
    for i in range(len(image)):
        total_brightness += image[i]
    return total_brightness / len(image)

index = 0

image = mnist.test.images[index]
label = mnist.test.labels[index]

conv_array_seperate = np.array(sess.run(separate_conv, feed_dict={x:image}))

conv_array_json = get_separate_conv_data(conv_array_seperate,10)

print(conv_array_json['separate_conv1'])

#print("Actual=", mnist.test.labels[index].argmax(), ": Prediction=", features[1])

#print(np.array(y1).shape)

#print(np.absolute(y2[0][0].squeeze()))

#fig = plt.figure()
#fig.add_subplot(111)
#plt.imshow(np.absolute(y1[0][0].squeeze()), cmap='gray')
#plt.axis('off')
#for j in range(32):
#    fig.add_subplot(1,33,j + 1)
#    plt.imshow(y2[2][j].squeeze(), cmap='gray')
#    plt.axis('off')
#for j in range(32):
#    fig.add_subplot(2,33,j + 1)
#    plt.imshow(features_pool1[j].squeeze(), cmap='gray')
#    plt.axis('off')
#for j in range(64):
#    fig.add_subplot(3,65,j + 1)
#    plt.imshow(features_conv2[j].squeeze(), cmap='gray')
#    plt.axis('off')
#for j in range(64):
#    fig.add_subplot(4,65,j + 1)
#    plt.imshow(features_pool2[j].squeeze(), cmap='gray')
#    plt.axis('off')
#plt.show()
#print("test accuracy %g"%accuracy.eval(feed_dict={
#    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
