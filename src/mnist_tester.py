import tensorflow as tf
import random
import numpy as np
import matplotlib.pyplot as plt
import math
import json
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('resource/MNIST_data', one_hot=True)

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def get_feature_map(layer, image_size, channels):
    temp_image = layer.reshape((image_size, image_size, channels))
    temp_image = temp_image.transpose((2, 0, 1))
    return temp_image.reshape((-1, image_size, image_size, 1))

def get_image_brightness(image):
    total_brightness = 0
    for i in range(len(image)):
        total_brightness += image[i]
    return total_brightness / len(image)

def get_feature_json(features): # takes in contents of different layers in CNN e.g. conv1 or pool2
    feature_list = []
    feature_brightness = []
    for i in range(len(features)):
        feature_data = {}
        feature_data['feature_' + str(i)] = to_three_channels(features[i].tolist())
        feature_brightness.append(get_image_brightness(features[i].flatten()))
        feature_list.append(feature_data)
    feature_list.append(feature_brightness)
    return feature_list

def to_three_channels(images): # takes in the images contained inside a feature
    image_list = []
    for i in range(len(images)):
        for j in range(len(images[i])):
            for k in range(3):
                image_list.append(images[i][j])
            image_list.append(255)
    return image_list

def convolution(image, label):
    x = tf.placeholder(tf.float32, shape=[784])
    y_ = tf.placeholder(tf.float32, shape=[10])

    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    x_image = tf.reshape(x, [-1,28,28,1])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    fc_decision_data = tf.nn.softmax(y_conv)

    saver = tf.train.Saver()

    sess = tf.Session()

    saver.restore(sess, "pre-trained/mnist/graph/mnist.ckpt")

    decision = fc_decision_data.eval(feed_dict={x: image, y_: label,
            keep_prob: 1.0})

    features_image = np.round(np.multiply(x_image.eval(feed_dict={x: image, y_: label,
            keep_prob: 1.0}), 255), decimals=0).squeeze()

    features_conv1 = np.round(np.multiply(get_feature_map(h_conv1.eval(feed_dict={x: image, y_: label,
            keep_prob: 1.0}), 28, 32), 255), decimals=0).squeeze()

    features_pool1 = np.round(np.multiply(get_feature_map(h_pool1.eval(feed_dict={x: image, y_: label,
            keep_prob: 1.0}), 14, 32), 255), decimals=0).squeeze()

    features_conv2 = np.round(np.multiply(get_feature_map(h_conv2.eval(feed_dict={x: image, y_: label,
            keep_prob: 1.0}), 14, 64), 255), decimals=0).squeeze()

    features_pool2 = np.round(np.multiply(get_feature_map(h_pool2.eval(feed_dict={x: image, y_: label,
            keep_prob: 1.0}), 7, 64), 255), decimals=0).squeeze()

    fully_con1 = np.round(np.multiply(h_fc1.eval(feed_dict={x: image, y_: label,
            keep_prob: 1.0}), 255), decimals=0).reshape([32,32]).squeeze()

    features = {}
    features[1] = np.array([features_image.tolist()])
        #features[2] = get_feature_json(features_conv1)
        #features[3] = get_feature_json(features_pool1)
        #features[4] = get_feature_json(features_conv2)
        #features[5] = get_feature_json(features_pool2)
        #features[6] = get_feature_json(np.array([fully_con1.tolist()]))

    data = {}
    data['features'] = features
        #data['prediction'] = decision.argmax().squeeze().tolist()
        #data['certainty'] = np.round(np.multiply(decision,100.0).squeeze(),decimals=8).tolist()

    return data

index = random.randint(0,10000)
features = convolution(mnist.test.images[index], mnist.test.labels[index])

#print features

#text_file = open("Output.txt", "w")
#text_file.write(features)
#text_file.close()

#print("Actual=", mnist.test.labels[index].argmax(), ": Prediction=", features[1])

#print(features[1])

#fig = plt.figure()
#fig.add_subplot(111)
#plt.imshow(features[0], cmap='gray')
#plt.axis('off')

#for j in range(32):
#    fig.add_subplot(1,33,j + 1)
#    plt.imshow(features_conv1[j].squeeze(), cmap='gray')
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
