from __future__ import print_function
import tensorflow as tf
from flask import Flask, render_template, request, send_file
from tensorflow.examples.tutorials.mnist import input_data
import random
import numpy as np
import json
import network_json
import sys
#from matplotlib import pyplot as plt

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
conv_array = np.array(sess.run(separate_conv, feed_dict={x:image}))

def get_separate_conv_data(data):
    separate = {}
    #np.round(np.multiply(get_feature_map(feature_list[1], 28, 32), 255), decimals=0)
    sep_1 = np.array([np.array(data[0]).transpose((2,5,0,1,3,4)).squeeze().tolist()])
    sep_2 = np.array(data[1]).transpose((2,5,0,1,3,4)).squeeze()
    separate['separate_conv1'] = get_highest_layer_activations(5,sep_1)
    separate['separate_conv2'] = get_highest_layer_activations(5,sep_2)#np.array(data[1]).transpose((2,5,0,1,3,4)).squeeze().shape()
    return separate, brightness

def get_highest_layer_activations(threshold, data):
    brightness = []
    for i in range(len(data)):
        feature_brightness = []
        for j in range(len(data[i])):
            temp = get_image_brightness(np.array(data[i][j]).flatten())
            feature_brightness.append(temp)
        brightness.append(feature_brightness)
    total_brightness = []
    brightness_copy = brightness
    for i in range(len(brightness)):
        top_feature_brightness = []
        for j in range(threshold):
            maximum = max(brightness[i])
            index = brightness[i].index(maximum)
            top_feature_brightness.append([maximum, i, index])
            brightness[i][index] = 0
        total_brightness.append(top_feature_brightness)
    return total_brightness, brightness_copy

def get_image_brightness(image):
    total_brightness = 0
    for i in range(len(image)):
        total_brightness += image[i]
    return total_brightness / len(image)

conv_array_json, brightness = get_separate_conv_data(conv_array)

print(brightness)
print(conv_array_json['separate_conv1'])
