from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import os
import inspect
import classifier
import sys
import argparse
import datasets
import shutil

def check_range(value):
    fvalue = float(value)
    if fvalue < 0.0 or fvalue > 1.0:
        raise argparse.ArgumentTypeError("Enter a value between 0 and 1")
    return fvalue

def check_dataset(value):
    svalue = str(value)
    if svalue != 'mnist' and svalue != 'cifar':
        raise argparse.ArgumentTypeError("Enter a valid dataset. (mnist/cifar)")
    return svalue

def check_pos(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("Enter a number greater than 0.")
    return ivalue

parser = argparse.ArgumentParser(description='ConvNet trainer usage.')
parser.add_argument('dataset', metavar='D', type=check_dataset, help='The name of the dataset. (mnist/cifar)')
parser.add_argument('iterations', metavar='I', type=check_pos, help='The number of training iterations')
parser.add_argument('batch_size', metavar='B', type=check_pos, help='A value for the number of batches used to train the network')
parser.add_argument('keep_prob', metavar='K', type=check_range, help='Probability of keeping a node active during dropout')

args = parser.parse_args()

#Extracting commandline arguments from args. namespace
dataset = args.dataset
iterations = args.iterations
batch_size = args.batch_size
keep_prob_args = args.keep_prob

path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) + "/pre-trained/" + dataset
if os.path.exists(path):
    choice = raw_input(dataset + "save already exists\nDo you want to overwrite previous " + dataset + " data? (y/n):")
    if choice[0] == 'y':
        shutil.rmtree(path)
    else:
        quit()

#Checking if dataset is cifar or mnist
if dataset == 'cifar': #Imports file to parse cifar-10
    loaded_data = datasets.CIFAR_Data()
elif dataset == 'mnist':
    loaded_data = datasets.MNIST_Data()

test_data = loaded_data.test_dataset

train_data = loaded_data.train_dataset

image_dimensions = loaded_data.image_dimensions

if batch_size > len(train_data.images):
    print("Batch data exceeds size of training data. Enter a value smaller than " + str(len(train_data.images)))
    quit()

#Put variables from classifier.py's variable scope "conv" into current session
#Only y_conv and variables are required for training
with tf.variable_scope("conv"):
    x = tf.placeholder(tf.float32, [None, image_dimensions * image_dimensions])
    y_ = tf.placeholder(tf.float32, [None,10])
    keep_prob = tf.placeholder(tf.float32)
    y_conv, variables, _, _ = classifier.conv(x,image_dimensions,keep_prob)

#Cross entropy is a value generated to aid in minimising the loss function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
#Trains using a stochastic optimiser: Adam
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#Checks if prediction and actual label are the same
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#Creates a tensorflow saver
#Allows the graph to be saved to a checkpoint file, with a max of 16 per session
saver = tf.train.Saver(variables, max_to_keep = 10000)
init_op = tf.initialize_all_variables()

#Initialises the tensorflow session as sess
with tf.Session() as sess:
    #All variables are now initialised and ready for input
    sess.run(init_op)
    for i in range(iterations + 1):
        #Obtaining the batch from the DataSet object
        batch = loaded_data.train_dataset.next_batch(batch_size)
        #Every 50 iterations the accuracy for that batch is printed and the graph saved to a checkpoint
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x:batch[0], y_: batch[1], keep_prob: keep_prob_args})
            print("%d:%g"%(i, train_accuracy))
            #print("Iteration:%d Accuracy:%g"%(i, train_accuracy))
            path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) + "/pre-trained/" + dataset + "/graph_" + dataset + str(i)
            if not os.path.exists(path):
                os.makedirs(path)
            saver.save(sess, "" + path + "/" + dataset + ".ckpt")
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: keep_prob_args})
    print("test accuracy %g"%accuracy.eval(feed_dict={
        x: test_data.images, y_: test_data.labels, keep_prob: 1.0}))
