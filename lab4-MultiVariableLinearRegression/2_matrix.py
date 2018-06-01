import tensorflow as tf

x_data = [
    [73., 80., 75.],
    [93., 88., 93.],
    [89., 91., 90.],
    [96., 98., 100.],
    [73., 66., 70.]
]

y_data = [
    [152.],
    [185.],
    [180.],
    [196.],
    [142.]
]

X = tf.placeholder(tf.float32, shape=[None, 3])
# shape: [# of train data , # of features]
Y = tf.placeholder(tf.float32, shape=[None, 1])
# [None,1] 1 = y

W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(X, W) + b

# cost
cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

for step in range(2001):
    costVal, hyVal, _ = sess.run([cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})
    if step % 10 == 0:
        print(step, "Cost: ", costVal, "\nPrediction:\n", hyVal, "\nTest:\n", _)
