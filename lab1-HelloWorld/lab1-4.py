# Basic Operations with variable as graph input
# The value returned by the constructor represents the output of the variable op. (define as input when running session)
# tf Graph input

import tensorflow as tf

a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

# 데이터 타입만 정해져있는 모델
# Define some operations == 모델
add = tf.add(a, b)
mul = tf.multiply(a, b)

with tf.Session() as sess:
    print("Addition with variable: %i" % sess.run(add, feed_dict={a: 2, b: 3}))
    print("Multiplication with variables: %i" % sess.run(mul, feed_dict={a: 2, b: 3}))
