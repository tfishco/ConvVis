from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import os
import inspect
import classifier
import sys

if len(sys.argv) == 3:
    if(sys.argv[1] != 'mnist' and sys.argv[1] != 'cifar'):
        print("Please use 'mnist' or 'cifar' in dataset selection")
        quit()
    else:
        dataset = sys.argv[1]
    iterations = int(sys.argv[2]) + 1
else:
    print("Please use: sudo python app.py <dataset (mnist/cifar)> <training iterations>")
    quit()

batch_number = 50

if dataset == 'cifar':
    import cifar_10
    cifar_10.load_and_preprocess_input(dataset_dir='resource/CIFAR_data')
    image_dimensions = cifar_10.image_width
    test_data = cifar_10.validate_all['data'][:,:,:,None,1].reshape(-1,image_dimensions * image_dimensions)
    test_labels = cifar_10.validate_all['labels']
    image_dimensions = cifar_10.image_width
    cifar_10.batch_size = batch_number

elif dataset == 'mnist':
    mnist = input_data.read_data_sets('/src/resource/MNIST_data', one_hot=True)
    test_data = mnist.test.images
    test_labels = mnist.test.labels
    image_dimensions = 28


sess = tf.Session()

with tf.variable_scope("conv"):
    x = tf.placeholder(tf.float32, [None, image_dimensions * image_dimensions])
    keep_prob = tf.placeholder(tf.float32)
    y_conv, variables, _ , _= classifier.conv(x,keep_prob,image_dimensions)

y_ = tf.placeholder(tf.float32, [None,10])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver(variables, max_to_keep = 16)
init_op = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init_op)
    for i in range(iterations):
        if dataset=='cifar':
            batch = cifar_10.get_batch(i,cifar_10.train_all['data'], cifar_10.train_all['labels'])
        elif dataset =='mnist':
            batch = mnist.train.next_batch(batch_number)
        if i % 50 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x:batch[0], y_: batch[1], keep_prob: 1.0})
            print("%d:%g"%(i, train_accuracy))
            path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) + "/pre-trained/" + dataset + "/graph_" + dataset + str(i)
            if not os.path.exists(path):
                os.makedirs(path)
            saver.save(sess, "" + path + "/" + dataset + ".ckpt")
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    print("test accuracy %g"%accuracy.eval(feed_dict={
        x: test_data, y_: test_labels, keep_prob: 1.0}))
