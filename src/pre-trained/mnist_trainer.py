from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import os
import inspect
import classifier

mnist = input_data.read_data_sets('/src/resource/MNIST_data', one_hot=True)

sess = tf.Session()

with tf.variable_scope("conv"):
    x = tf.placeholder(tf.float32, [None, 784])
    keep_prob = tf.placeholder(tf.float32)
    y_conv, variables, _ , _= classifier.conv(x,keep_prob)

y_ = tf.placeholder(tf.float32, [None,10])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver(variables, max_to_keep = 11)
init_op = tf.initialize_all_variables()

iterations = 101

with tf.Session() as sess:
    sess.run(init_op)
    for i in range(iterations):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x:batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g"%(i, train_accuracy))
            path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) + "/mnist/graph_mnist" + str(i)
            if not os.path.exists(path):
                os.makedirs(path)
            saver.save(sess, "" + path + "/mnist.ckpt")
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    print("test accuracy %g"%accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
