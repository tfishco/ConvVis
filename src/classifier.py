import tensorflow as tf

def conv(x, keep_prob, image_dimensions):
    def weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)

    def bias_variable(shape):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)

    def separate_conv(x,W):
        convolutions = []
        for i in range(W[:].get_shape()[2]):
            single_x = x[:,:,:,None,i]
            image_W = W[:,:,None,i,:]
            new = []
            for j in range(W[:].get_shape()[3]):
                single_W = image_W[:,:,:,None,j]
                conv = tf.nn.relu(conv2d(single_x, single_W))
                new.append(conv)
            convolutions.append(new)
        return convolutions


    def conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')

    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    x_image = tf.reshape(x, [-1,image_dimensions,image_dimensions,1])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_conv1_separate = separate_conv(x_image, W_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_conv2_separate = separate_conv(h_pool1, W_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([(image_dimensions / 4) * (image_dimensions / 4) * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, (image_dimensions / 4) * (image_dimensions / 4) * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    fc_decision_data = tf.nn.softmax(y_conv)

    return y_conv, [W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2], [x_image, h_conv1, h_pool1, h_conv2, h_pool2, h_pool2_flat, h_fc1, h_fc1_drop, fc_decision_data] , [h_conv1_separate, h_conv2_separate]
