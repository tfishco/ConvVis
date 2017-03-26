import tensorflow as tf

def conv(x, keep_prob, image_dim, dropout):
	""" A method used to obtain variables contained within the ConvNet as well
	as to feed variables into the network.

	Parameters
	----------
	x : A tensorflow placeholder for image input
	keep_prob : The probability of keeping a node active in dropout layer
	image_dim : The width and assumed height of x
	dropout : Dropout toggle

	Returns
	-------
	All variables contained in the function in the format:
		prediction,[variables],[conv featuremaps],[seperated convolutions]
	"""
	def weight_variable(shape):
		"""Method for creating a standard tensorflow variable

		Parameters
		----------
		shape : Array of length 4

		Returns
		-------
		A variable of the given shape
		"""
		initial = tf.truncated_normal(shape, stddev=0.1)
		return tf.Variable(initial)

	def bias_variable(shape):
		"""Used to create a variable of predetermined value (0.1)

		Parameters
		----------
		shape : Array of length 4

		Returns
		-------
		A variable of the given shape
		"""
		initial = tf.constant(0.1, shape=shape)
		return tf.Variable(initial)

	def separate_conv(x, W):
		"""Seperates the convolution process through slicing all featuremaps in
		the previous layer / dimension

		Parameters
		----------
		x : The input volume, shape = [Volume Depth, Image Width, Image Height,
										Consequent Volume Depth]
		W : The weights volume, shape = [Image Width, Image Height,
							Antecedent Volume Depth, Consequent Volume Depth]

		Returns
		-------
		Array of expected shape if it were to be convolved without
		slicing
		"""
		convolutions = []
		#Iterates through the third dimension of the weight volume, since the
		#this dimension is the antecedent layer volume depths
		for i in range(W[:].get_shape()[3]):
			#Slices input volume x along its fourth dimension, using None types
			#to maintain original dimenions and obtaining a volume of features
			#Slices the weight volume along its second dimension, to obtain a
			#volume of weights of the same depth .
			image_W = W[:,:,:,None,i]
			new = []
			#Iterates through W's fourth dimension,
			for j in range(W[:].get_shape()[2]):
				single_x = x[:,:,:,None,j]
				#Obtains a single weights matrix
				single_W = image_W[:,:,None,j,:]
				#Convolves this matrix with the volume from single_x
				conv = tf.nn.relu(conv2d(single_x, single_W))
				new.append(conv)
			convolutions.append(new)
		return convolutions


	def conv2d(x, W):
		"""The convolution layer

		Parameters
		----------
		x : The input volume
		W : The weights volume

		Returns
		-------
		Volume of all colvolved features
		"""
		return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

	def max_pool_2x2(x):
		"""Performs a 2x2 max pooling operation

		Parameters
		----------
		x : The input volume

		Returns
		-------
		Sub-sampled volume
		"""
		return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
							strides=[1, 2, 2, 1], padding='SAME')

	W_conv1 = weight_variable([5, 5, 1, 32])
	b_conv1 = bias_variable([32])

	x_image = tf.reshape(x, [-1,image_dim,image_dim,1])

	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	h_conv1_separate = separate_conv(x_image, W_conv1)
	h_pool1 = max_pool_2x2(h_conv1)

	W_conv2 = weight_variable([5, 5, 32, 64])
	b_conv2 = bias_variable([64])

	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	h_conv2_separate = separate_conv(h_pool1, W_conv2)
	h_pool2 = max_pool_2x2(h_conv2)

	W_fc1 = weight_variable([(image_dim / 4) * (image_dim / 4) * 64, 1024])
	b_fc1 = bias_variable([1024])

	h_pool2_flat = tf.reshape(h_pool2, [-1, (image_dim / 4) * (image_dim / 4) * 64])

	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

	if dropout:
		h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
	else:
		h_fc1_drop = h_fc1

	W_fc2 = weight_variable([1024, 10])
	b_fc2 = bias_variable([10])

	y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

	fc_decision_data = tf.nn.softmax(y_conv)

	return y_conv, [W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2], [x_image, h_conv1, h_pool1, h_conv2, h_pool2, h_pool2_flat, h_fc1, h_fc1_drop, fc_decision_data] , [h_conv1_separate, h_conv2_separate]
