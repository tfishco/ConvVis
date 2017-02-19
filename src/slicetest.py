import tensorflow as tf
import numpy as np

def fun_fun1():
    x = tf.placeholder("int32")
    i = tf.placeholder("int32")
    y = tf.slice(x,[0,0,0],[-1,-1,2])

    #initialize
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    x_ = [[[1,2,3],[4,5,6],[7,8,9]],[[10,11,12],[13,14,15],[16,17,18]],[[19,20,21],[22,23,24],[25,26,27]]]

    #run
    result = sess.run(y, feed_dict={x:x_})
    print(result)

def fun_fun2():
    x = tf.placeholder("float")
    i = tf.placeholder("int32")
    y = tf.reverse(x, [False, False, True])

    #initialize
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    x_ = [[[1,2,3],[4,5,6],[7,8,9]],[[10,11,12],[13,14,15],[16,17,18]],[[19,20,21],[22,23,24],[25,26,27]]]

    #run
    result = sess.run(y, feed_dict={x:x_})
    print(result)

fun_fun2()
