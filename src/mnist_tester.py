import tensorflow as tf
import random
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

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

def convolution(image, label):
    input_features = [None] * 7
    sess = tf.InteractiveSession()

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

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()
    saver.restore(sess, "pre-trained/mnist/checkpoint/mnist.ckpt")

    decision = fc_decision_data.eval(feed_dict={x: image, y_: label,
        keep_prob: 1.0})

    #print("Actual : Predicition")
    #print(mnist.test.labels[index].argmax(), ":", decision.argmax())

    features_image = x_image.eval(feed_dict={x: image, y_: label,
        keep_prob: 1.0})

    features_conv1 = get_feature_map(h_conv1.eval(feed_dict={x: image, y_: label,
        keep_prob: 1.0}), 28, 32)

    features_pool1 = get_feature_map(h_pool1.eval(feed_dict={x: image, y_: label,
        keep_prob: 1.0}), 14, 32)

    features_conv2 = get_feature_map(h_conv2.eval(feed_dict={x: image, y_: label,
        keep_prob: 1.0}), 14, 64)

    features_pool2 = get_feature_map(h_pool2.eval(feed_dict={x: image, y_: label,
        keep_prob: 1.0}), 7, 64)

    input_features[0] = features_image.squeeze()
    input_features[1] = features_conv1.squeeze()
    input_features[2] = features_pool1.squeeze()
    input_features[3] = features_conv2.squeeze()
    input_features[4] = features_pool2.squeeze()
    input_features[5] = decision.argmax().squeeze()
    input_features[6] = decision.squeeze()

    return input_features

#index = random.randint(0,10000)
#features = convolution(mnist.test.images[index], mnist.test.labels[index])
#
#print("Actual=", mnist.test.labels[index].argmax(), ": Prediction=", features[5])
#
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
plt.show()

#print("test accuracy %g"%accuracy.eval(feed_dict={
#    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
