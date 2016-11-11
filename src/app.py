from __future__ import print_function # In python 2.7
from flask import Flask, render_template, request, send_file
from tensorflow.examples.tutorials.mnist import input_data
import sys
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import time

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

###############################################################################
################################# Conv ########################################
###############################################################################
def convnet_main(iteration):

    sess = tf.InteractiveSession()

    x = tf.placeholder(tf.float32, shape=[None, 28*28])
    y_ = tf.placeholder(tf.float32, shape=[None,10])

    W = tf.Variable(tf.zeros([28*28,10]))
    b = tf.Variable(tf.zeros([10]))

    sess.run(tf.initialize_all_variables())

    y = tf.matmul(x,W) + b

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y,y_))

    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    def weight_variable(shape):
        inital = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(inital)

    def bias_variable(shape):
        inital = tf.constant(0.1, shape=shape)
        return tf.Variable(inital)

    def conv2d(x,W):
        return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding="SAME")

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

    x_image = tf.reshape(x, [-1,28,28,1])

    W_conv1 = weight_variable([5,5,1,32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5,5,32,64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024,10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sess.run(tf.initialize_all_variables())
    for i in range(0, iteration):
      batch = mnist.train.next_batch(50)
      train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    return accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})


################################################################################
################################ Flask #########################################
################################################################################


app = Flask(__name__)

@app.route("/")
def index():
    return render_template("main.html")

@app.route("/train", methods=["POST"])
def train():
    iteration = int(request.form['iter'])
    start_time = time.time()
    accuracy = convnet_main(iteration)
    end_time = time.time() - start_time
    return render_template("train.html",accuracy=accuracy, time=round(end_time,1))

@app.route("/ga")
def ga():
    if request.args.get("gene_size")==None:
        gene_size=26
    else:
        gene_size = int(request.args.get("gene_size"))

    if request.args.get("population_size")==None:
        population_size=26
    else:
        population_size = int(request.args.get("population_size"))

    if request.args.get("generations")==None:
        generations=26
    else:
        generations = int(request.args.get("generations"))

    if request.args.get("mutation_rate")==None:
        mutation_rate=0.1
    else:
        mutation_rate = float(request.args.get("mutation_rate"))

    fitnessData = GAMain(gene_size, population_size, generations, mutation_rate)
    max = 100
    return render_template("main.html",fitnessData=fitnessData,max=max)

if __name__ == "__main__":
    app.debug = True
    app.run()
