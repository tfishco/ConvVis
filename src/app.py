import tensorflow as tf
from flask import Flask, render_template, request, send_file
from tensorflow.examples.tutorials.mnist import input_data
import random
import math
import numpy as np
import json
import network_json0 as network_json
import sys

sys.path.insert(0, 'pre-trained/')

import classifier

mnist = input_data.read_data_sets('resource/MNIST_data', one_hot=True)

def get_image_brightness(image):
    total_brightness = 0
    for i in range(len(image)):
        total_brightness += image[i]
    return total_brightness / len(image)

def get_feature_json(features): # takes in contents of different layers in CNN e.g. conv1 or pool2
    feature_list = []
    feature_brightness = []
    max_brightness = 0
    for i in range(len(features)):
        feature_data = {}
        feature_data['feature_' + str(i)] = to_rgba(features[i].tolist())
        image_brightness = get_image_brightness(features[i].flatten())

        if image_brightness > max_brightness:
            max_brightness = image_brightness

        feature_brightness.append(image_brightness)
        feature_list.append(feature_data)
    feature_list.append(feature_brightness)
    return feature_list, max_brightness

def to_rgba(images): # takes in the images contained inside a feature
    image_list = []
    for i in range(len(images)):
        for j in range(len(images[i])):
            for k in range(3):
                image_list.append(images[i][j])
            image_list.append(255)
    return image_list

def get_feature_map(layer, image_size, channels):
    temp_image = layer.reshape((image_size, image_size, channels))
    temp_image = temp_image.transpose((2, 0, 1))
    return temp_image.reshape((-1, image_size, image_size, 1))

def get_conv_data(feature_list):
    features = {}
    features['1'], max1 = get_feature_json(np.array(np.round(np.multiply(feature_list[0], 255), decimals=0).tolist()))
    features['2'], max2 = get_feature_json(np.round(np.multiply(get_feature_map(feature_list[1], 28, 32), 255), decimals=0))
    features['3'], max3 = get_feature_json(np.round(np.multiply(get_feature_map(feature_list[2], 14, 32), 255), decimals=0))
    features['4'], max4 = get_feature_json(np.round(np.multiply(get_feature_map(feature_list[3], 14, 64), 255), decimals=0))
    features['5'], max5 = get_feature_json(np.round(np.multiply(get_feature_map(feature_list[4], 7, 64), 255), decimals=0))
    features['6'], max6 = get_feature_json([np.round(np.multiply(feature_list[6], 255), decimals=0)])

    data = {}
    data['features'] = features
    data['prediction'] = np.argmax(feature_list[8])
    data['certainty'] = np.round(np.multiply(feature_list[8],100.0).squeeze(),decimals=8).tolist()
    data['log_certainty'] = np.log1p(np.array(feature_list[8])).squeeze().tolist()
    data['max_brightness'] = np.amax([max1,max2,max3,max4,max5,max6]).tolist()
    return data

def get_weight_data(weights_list):
    data = {}
    data['fc1'] = get_fc1_sum(weights_list[4].tolist(), 7*7)
    return data

def get_fc1_sum(weight_list, area):
    pixel_weights = []
    data = {}
    for i in range(0,len(weight_list),area):
        image_weights = []
        for j in range(area):
            image_weights.append(np.sum(np.array(weight_list[i + j])))
        pixel_weights.append(np.sum(np.array(image_weights)))
    data['abs'] = np.divide(np.absolute(np.array(pixel_weights)),8).tolist()
    data['raw'] = np.divide(np.array(pixel_weights),8).tolist()

    return data

def get_separate_conv_data(data, threshold):
    separate = {}
    #np.round(np.multiply(get_feature_map(feature_list[1], 28, 32), 255), decimals=0)
    sep_1 = np.array([np.array(data[0]).transpose((2,5,0,1,3,4)).squeeze().tolist()])
    sep_2 = np.array(data[1]).transpose((2,5,0,1,3,4)).squeeze()
    separate['separate_conv1'], separate['indexes_conv1'] = get_highest_layer_activations(threshold,sep_1)
    separate['separate_conv2'], separate['indexes_conv2'] = get_highest_layer_activations(threshold,sep_2)#np.array(data[1]).transpose((2,5,0,1,3,4)).squeeze().shape()
    return separate

def get_highest_layer_activations(threshold, data):
    brightness = []
    for i in range(len(data)):
        feature_brightness = []
        for j in range(len(data[i])):
            temp = get_image_brightness(np.multiply(np.array(data[i][j]).flatten(), 255))
            feature_brightness.append(temp)
        brightness.append(feature_brightness)
    total_brightness = []
    for i in range(len(brightness)):
        total_brightness.append(np.array(brightness[i]).argsort()[-threshold:][::-1].tolist())
    return brightness, total_brightness

x = tf.placeholder("float", [784])

sess = tf.Session()

train_iteration = 0

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

with tf.variable_scope("conv"):
    _, variables, features , separated_conv = classifier.conv(x, 1.0)
saver = tf.train.Saver(variables)
saver.restore(sess, "pre-trained/mnist/graph_mnist" + str(train_iteration) + "/mnist.ckpt")


@app.route("/conv", methods=['POST'])
def conv():

    index = int(request.form['val'])
    struct = json.loads(request.form['struct'])
    image = mnist.test.images[index]
    label = mnist.test.labels[index]

    data = {}
    data['label'] = np.argmax(label)
    data['weightdata'] = get_weight_data(sess.run(variables, feed_dict={x:image}))
    data['convdata'] = get_conv_data(sess.run(features, feed_dict={x:image}))
    separated_conv_data = get_separate_conv_data(sess.run(separated_conv, feed_dict={x:image}), 10)
    data['separated_convdata'] = separated_conv_data
    data['struct'], data['no_nodes'] = network_json.get_json(struct[0], struct[1], data['convdata']['log_certainty'], separated_conv_data)
    return json.dumps(data)

if __name__ == "__main__":
    app.run()
