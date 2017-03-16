from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import os
import inspect
import classifier
import cifar_10

#mnist = input_data.read_data_sets('/src/resource/MNIST_data', one_hot=True)

dataset ='cifar'
#dataset = 'mnist'

if dataset == 'cifar':
    image_dimensions = cifar_10.image_width
    cifar_10.load_and_preprocess_input(dataset_dir='resource/CIFAR_data')
    test_data = cifar_10.validate_all['data'][:,:,:,None,1]
    test_labels = cifar_10.validate_all['labels']
    image_dimensions = cifar_10.image_width

elif dataset == 'mnist':
    test_data = mnist.test.images
    test_labels = mnist.test.labels
    image_dimensions = 28

cifar_10.load_and_preprocess_input(dataset_dir=cifar_10.dataset_dir)

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

saver = tf.train.Saver(variables, max_to_keep = 11)
init_op = tf.initialize_all_variables()

iterations = 201

with tf.Session() as sess:
    sess.run(init_op)
    for i in range(iterations):
        if dataset=='cifar':
            batch = cifar_10.get_batch(i,cifar_10.train_all['data'], cifar_10.train_all['labels'])
        elif dataset =='mnist':
            batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x:batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g"%(i, train_accuracy))
            path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) + "/" + dataset + "/graph_" + dataset + str(i)
            if not os.path.exists(path):
                os.makedirs(path)
            saver.save(sess, "" + path + "/" + dataset + ".ckpt")
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    print("test accuracy %g"%accuracy.eval(feed_dict={
        x: test_data, y_: test_labels, keep_prob: 1.0}))
