from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

sess = tf.InteractiveSession()

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

def get_feature_map(layer, image_size, batch, channels):
    temp_image = layer[0].reshape((image_size, image_size, channels))
    temp_image = temp_image.transpose((2, 0, 1))
    return temp_image.reshape((-1, image_size, image_size, 1))

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

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

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

softmax = tf.nn.softmax_cross_entropy_with_logits(y_conv, y_)
cross_entropy = tf.reduce_mean(softmax)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init_op = tf.initialize_all_variables()
#saver = tf.train.Saver()

sess.run(init_op)
for i in range(1000):
  batch = mnist.train.next_batch(50)

  decision = softmax.eval(feed_dict={x: batch[0], y_: batch[1],
    keep_prob: 1.0})
  print(decision)

  featuresc1 = get_feature_map(h_conv1.eval(feed_dict={x: batch[0], y_: batch[1],
    keep_prob: 1.0}), 28, batch, 32)

  featuresc2 = get_feature_map(h_conv2.eval(feed_dict={x: batch[0], y_: batch[1],
    keep_prob: 1.0}), 14, batch, 64)

  fig = plt.figure()
  for j in range(32):
      fig.add_subplot(1,33,j + 1)
      plt.imshow(featuresc1[j].squeeze(), cmap='gray')
      plt.axis('off')
  for j in range(64):
      fig.add_subplot(2,65,j + 1)
      plt.imshow(featuresc2[j].squeeze(), cmap='gray')
      plt.axis('off')
  plt.show()



  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
#saver.save(sess, "pre-trained/mnist/mnist.ckpt")
