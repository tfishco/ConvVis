from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import os
import inspect
import classifier
import sys
import argparse

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
    if dataset == 'cifar':
        import cifar_10 #Imports file to parse cifar-10
        cifar_10.load_and_preprocess_input(dataset_dir='resource/CIFAR_data')
        image_dimensions = cifar_10.image_width
        test_data = cifar_10.validate_all['data'][:,:,:,None,1].reshape(-1,image_dimensions * image_dimensions)
        test_labels = cifar_10.validate_all['labels']
        train_data = cifar_10.train_all['data']
        train_labels = cifar_10.train_all['labels']
        image_dimensions = cifar_10.image_width
        cifar_10.batch_size = batch_number

    elif dataset == 'mnist':
        mnist = input_data.read_data_sets('/src/resource/MNIST_data', one_hot=True)
        test_data = mnist.test.images
        test_labels = mnist.test.labels
        image_dimensions = 28

#Put variables from classifier.py's variable scope "conv" into current session
#Only y_conv and variables are required for training
with tf.variable_scope("conv"):
    x = tf.placeholder(tf.float32, [None, image_dimensions * image_dimensions])
    y_ = tf.placeholder(tf.float32, [None,10])
    keep_prob = tf.placeholder(tf.float32)
    y_conv, variables, _ , _= classifier.conv(x,keep_prob,image_dimensions,dropout)

#Cross entropy is a value generated to aid in minimising the loss function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
#Trains using a stochastic optimiser: Adam
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#Checks if prediction and actual label are the same
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#Creates a tensorflow saver
#Allows the graph to be saved to a checkpoint file, with a max of 16 per session
saver = tf.train.Saver(variables, max_to_keep = 16)
init_op = tf.initialize_all_variables()

#Initialises the tensorflow session as sess
with tf.Session() as sess:
    #All variables are now initialised and ready for input
    sess.run(init_op)
    for i in range(iterations):
        #Obtaining the batch depending on what dataset is being used
        if dataset=='cifar':
            #The current cifar parser requires input of the iterations and data
            batch = cifar_10.get_batch(i,train_data, train_labels)
        elif dataset =='mnist':
            batch = mnist.train.next_batch(batch_number)
        #Every 100 iterations the accuracy for that batch is printed and the graph saved to a checkpoint
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x:batch[0], y_: batch[1], keep_prob: 1.0})
            print("Iteration:%d Accuracy:%g"%(i, train_accuracy))
            path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) + "/pre-trained/" + dataset + "/graph_" + dataset + str(i)
            if not os.path.exists(path):
                os.makedirs(path)
            saver.save(sess, "" + path + "/" + dataset + ".ckpt")
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    print("test accuracy %g"%accuracy.eval(feed_dict={
        x: test_data, y_: test_labels, keep_prob: 1.0}))
