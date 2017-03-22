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
import argparse

def get_image_brightness(image):
    """Gets the total pixel brightness for an array of pixels"""
    total_brightness = 0
    for i in range(len(image)):
        total_brightness += image[i]
    return total_brightness

def get_features(features): # takes in contents of different layers in CNN e.g. conv1 or pool2
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
    """Converts array of single channel image into four channel by replicating
    the value and adding 255 for alpha.
    """
    image_list = []
    for i in range(len(images)):
        for j in range(len(images[i])):
            for k in range(3):
                image_list.append(images[i][j])
            image_list.append(255)
    return image_list

def get_feature_map(layer, image_size, channels):
    """Transposes the volume into a simpler format to parse"""
    temp_image = layer.reshape((image_size, image_size, channels))
    temp_image = temp_image.transpose((2, 0, 1))
    return temp_image.reshape((-1, image_size, image_size, 1))

def get_conv_data(feature_list):
    """Creates a dictionary to represent the ConvNet. JSON of form :
                                                {features: [...], data: [...]}
    Features contains pixel values for convolution layers, Data contains other
    various metrics
    """
    features = {}
    features['1'], max1 = get_features(np.array(np.round(np.multiply(feature_list[0], 255), decimals=0).tolist()))
    features['2'], max2 = get_features(np.round(np.multiply(get_feature_map(feature_list[1], image_dimensions, 32), 255), decimals=0))
    features['3'], max3 = get_features(np.round(np.multiply(get_feature_map(feature_list[2], image_dimensions / 2, 32), 255), decimals=0))
    features['4'], max4 = get_features(np.round(np.multiply(get_feature_map(feature_list[3], image_dimensions / 2, 64), 255), decimals=0))
    features['5'], max5 = get_features(np.round(np.multiply(get_feature_map(feature_list[4], image_dimensions / 4, 64), 255), decimals=0))
    features['6'], max6 = get_features([np.round(np.multiply(feature_list[6], 255), decimals=0)])

    data = {}
    data['features'] = features
    data['prediction'] = np.argmax(feature_list[8])
    data['certainty'] = np.round(np.multiply(feature_list[8],100.0).squeeze(),decimals=8).tolist()
    data['log_certainty'] = np.log1p(np.array(feature_list[8])).squeeze().tolist()
    data['max_brightness'] = np.amax([max1,max2,max3,max4,max5,max6]).tolist()
    return data

def get_weight_data(weights_list):
    print(len(weights_list[4]))
    data = {}
    data['fc1'] = get_fc1_sum(weights_list[4].tolist(), image_dimensions / 4 * image_dimensions / 4)
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
            indexes.append(data[i][j])
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
                    max_vals.append(max_indexes[k])
            else:
                prev_indexes = get_prev_index(top_brightness[i - 1])
                for k in range(threshold):
                    if j in prev_indexes:
                        max_vals.append(max_indexes[k])
            layers.append(max_vals)
        top_brightness.append(layers)
    return top_brightness

parser = argparse.ArgumentParser(description='ConvNet trainer usage.')
parser.add_argument('dataset', metavar='D', type=str, help='The name of the dataset. (mnist/cifar)')
parser.add_argument('iterations', metavar='I', type=int, help='The number of training iterations')
parser.add_argument('batch_number', metavar='B', type=int, help='A value for the number of batches used to train the network')
parser.add_argument('dropout', metavar='DR', type=bool, help='Toggle for dropout in/exclusion. (True/False)')

args = parser.parse_args()

#Extracting commandline arguments from args. namespace
dataset = args.dataset
iterations = args.iterations
dropout = args.dropout
batch_number = args.batch_number

#Checking if dataset is cifar or mnist
if(dataset != 'mnist' and dataset != 'cifar'):
    print("Please use 'mnist' or 'cifar' in dataset selection")
    quit()
else:
    if dataset == 'mnist':
        mnist = input_data.read_data_sets('resource/MNIST_data', one_hot=True)
        test_data = mnist.test.images
        test_labels = mnist.test.labels
        actual_class_labels = list(range(10))
        image_dimensions = 28

    if dataset == 'cifar':
        import cifar_10
        cifar_10.load_and_preprocess_input(dataset_dir='resource/CIFAR_data')
        image_dimensions = cifar_10.image_width
        test_data = cifar_10.validate_all['data'][:,:,:,None,1].reshape(-1,image_dimensions * image_dimensions)
        test_labels = cifar_10.validate_all['labels']
        actual_class_labels = cifar_10.actual_class_labels
        image_dimensions = cifar_10.image_width

sess = tf.Session()

with tf.variable_scope("conv"):
    x = tf.placeholder("float", [None,image_dimensions * image_dimensions])
    y_ = tf.placeholder(tf.float32, [None,10])
    keep_prob = tf.placeholder(tf.float32)
    y_conv, variables, features , separated_conv = classifier.conv(x, 1.0,image_dimensions)

saver = tf.train.Saver(variables)
saver.restore(sess, "pre-trained/" + dataset + "/graph_" + dataset + str(iterations) + "/" + dataset + ".ckpt")

def get_training(dataset):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    individual_test(test_data, test_labels)
    #train_acc = accuracy.eval(session=sess,feed_dict={
    #        x: test_data, y_: test_labels, keep_prob: 1.0})
    return 0

def individual_test(test_data, test_labels):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    test_prediction = sess.run(features, feed_dict={
            x: test_data, y_: test_labels, keep_prob: 1.0})
    print(np.array(test_prediction[7]).shape)

#Setting up the Flask container
app = Flask(__name__)

#Default route for the app
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/dataset_data", methods=['POST'])
def dataset_data():
    """Returns JSON containing information about loaded dataset"""
    print("hello");
    _dataset = request.form['name']
    data_json = {}
    accuracy = get_training(_dataset)
    data_json['test_accuracy'] = float(accuracy)
    return json.dumps(data_json)

@app.route("/conv", methods=['POST'])
def conv():
    """Returns JSON containing information about the selected test entry.
    Requires a 'val' and 'struct' from the requester.
    """
    #Recieved from post
    index = int(request.form['val'])
    struct = json.loads(request.form['struct'])
    #From test dataset
    image = test_data[index].reshape((-1,image_dimensions * image_dimensions))
    label = test_labels[index]
    #JSON Construction to be sent to front end
    data = {}
    data['label'] = np.argmax(label)
    data['actual_class_labels'] = actual_class_labels
    data['weightdata'] = get_weight_data(sess.run(variables, feed_dict={x:image}))
    data['convdata'] = get_conv_data(sess.run(features, feed_dict={x:image}))
    separated_conv_data = get_highest_layer_activations(10,get_separate_conv_data(sess.run(separated_conv, feed_dict={x:image})))
    data['training_iter'] = iterations;
    data['struct'], data['no_nodes'] = network_json.get_json(struct[0], struct[1], data['convdata']['log_certainty'], separated_conv_data)
    return json.dumps(data)

if __name__ == "__main__":
    app.run()
