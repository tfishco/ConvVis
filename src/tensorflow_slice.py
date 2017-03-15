import numpy as np
import tensorflow as tf


W = tf.Variable(tf.truncated_normal([5,5,1,5], stddev=0.1))

new_W = tf.transpose(W, perm=[3,2,0,1])

init_op = tf.initialize_all_variables()

#run the graph
with tf.Session() as sess:
    sess.run(init_op) #execute init_op
    #print the random values that we sample
    print (sess.run(W))
    print (sess.run(new_W))
