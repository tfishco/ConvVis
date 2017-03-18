import tensorflow as tf
from flask import Flask, render_template, request, send_file
from tensorflow.examples.tutorials.mnist import input_data
import random
import math
import numpy as np
import json
import network_json0 as network_json
import sys
import os

import classifier

if len(sys.argv) == 3:
    if(sys.argv[1] != 'mnist' and sys.argv[1] != 'cifar'):
        print("Please use 'mnist' or 'cifar' in dataset selection")
        quit()
    else:
        dataset = sys.argv[1]
    if os.path.isdir("pre-trained/" + dataset + "/graph_" + dataset + str(sys.argv[2])):
        iterations = sys.argv[2]
    else:
      print("Training iteration does not exist")
      quit()
else:
    print("Please use: sudo python app.py <dataset (mnist/cifar)> <training iterations>")
    quit()

if dataset == 'mnist':
    mnist = input_data.read_data_sets('resource/MNIST_data', one_hot=True)
    test_data = mnist.test.images
    test_labels = mnist.test.labels
    image_dimensions = 28

elif dataset == 'cifar':
    import cifar_10
    cifar_10.load_and_preprocess_input(dataset_dir='resource/CIFAR_data')
    test_data = cifar_10.validate_all['data'][:,:,:,None,1]
    test_labels = cifar_10.validate_all['labels']
    image_dimensions = cifar_10.image_width

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
    features['2'], max2 = get_feature_json(np.round(np.multiply(get_feature_map(feature_list[1], image_dimensions, 32), 255), decimals=0))
    features['3'], max3 = get_feature_json(np.round(np.multiply(get_feature_map(feature_list[2], image_dimensions / 2, 32), 255), decimals=0))
    features['4'], max4 = get_feature_json(np.round(np.multiply(get_feature_map(feature_list[3], image_dimensions / 2, 64), 255), decimals=0))
    features['5'], max5 = get_feature_json(np.round(np.multiply(get_feature_map(feature_list[4], image_dimensions / 4, 64), 255), decimals=0))
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
    data['fc1'] = get_fc1_sum(weights_list[4].tolist(), image_dimensions * image_dimensions)
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

def get_separate_conv_data(data):
    conv_layers = []
    carry_over = []
    for i in range(len(data)):                      # iterate through conv layers
        layer = np.array(data[i]).squeeze()
        if len(layer.shape) < 4:
            layer = np.array([layer.tolist()])
        activations = []
        for j in range(len(layer)):             # iterate through images in conv layers
            input_activations = np.array(layer[j]).squeeze()
            activation_brightnesses = []
            for k in range(len(input_activations)):    #iterate through image activations
                activation_brightness = get_image_brightness(input_activations[k].flatten())
                activation_brightnesses.append(activation_brightness)
            activations.append(activation_brightnesses)
        conv_layers.append(activations)
    return conv_layers

def get_prev_index(data):
    indexes = []
    for i in range(len(data)):
        for j in range(len(data[i])):
            indexes.append(data[i][j][1])
    return indexes

def get_highest_layer_activations(threshold, data):
    top_brightness = []
    for i in range(len(data)): #iterate through layers
        layers = []
        for j in range(len(data[i])): #iterate through images in layers
            max_indexes = np.array(data[i][j]).argsort()[-threshold:][::-1]
            max_vals = []
            if i < 1:
                for k in range(threshold):
                    max_vals.append([data[i][j][max_indexes[k]], max_indexes[k]])
            else:
                prev_indexes = get_prev_index(top_brightness[i - 1])
                for k in range(threshold):
                    if j in prev_indexes:
                        max_vals.append([data[i][j][max_indexes[k]], max_indexes[k]])
            layers.append(max_vals)
        top_brightness.append(layers)
    return top_brightness

x = tf.placeholder("float", [image_dimensions * image_dimensions])

sess = tf.Session()

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

with tf.variable_scope("conv"):
    _, variables, features , separated_conv = classifier.conv(x, 1.0,image_dimensions)
saver = tf.train.Saver(variables)
saver.restore(sess, "pre-trained/" + dataset + "/graph_" + dataset + str(iterations) + "/" + dataset + ".ckpt")


@app.route("/conv", methods=['POST'])
def conv():

    index = int(request.form['val'])
    struct = json.loads(request.form['struct'])
    image = test_data[index].reshape((image_dimensions * image_dimensions,))
    label = test_labels[index]

    data = {}
    data['label'] = np.argmax(label)
    data['weightdata'] = get_weight_data(sess.run(variables, feed_dict={x:image}))
    data['convdata'] = get_conv_data(sess.run(features, feed_dict={x:image}))
    separated_conv_data = get_highest_layer_activations(20,get_separate_conv_data(sess.run(separated_conv, feed_dict={x:image})))
    data['separated_convdata'] = separated_conv_data
    data['struct'], data['no_nodes'] = network_json.get_json(struct[0], struct[1], data['convdata']['log_certainty'], separated_conv_data)
    return json.dumps(data)

if __name__ == "__main__":
    app.run()
