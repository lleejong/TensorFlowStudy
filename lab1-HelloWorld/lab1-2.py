import tensorflow as tf

# Start tf Session
sess = tf.Session();

# Basic constant operations
# The value returned by the constructor represents the output of the Constant op.
a = tf.constant(2)
b = tf.constant(3)

c = a+b

#Print out operation
print(c)

print(sess.run(c))
