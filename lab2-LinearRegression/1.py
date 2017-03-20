import tensorflow as tf

x_data = [1, 2, 3]
y_data = [1, 2, 3]

#Try to find values for W and b that compute y_data = W * x_data + b
#(We know that W shoud be 1 and b 0, but Tensorflow will figure that out for us.)

#tf.Constant가 아닌 이유, training을 하면서 tf가 값을 적절히 변형 할 수 있도록 하기 위함
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

#Our hypothesis
hypothesis = W * x_data + b #H(x) = Wx + b

#Simplified cost function
cost = tf.reduce_mean(tf.square(hypothesis - y_data)) #cost(W,b) = 1/m * sigma(H(x) - y)^2
#실제로 계산이 이루어지진 않고, operation이 정의 됨

#Minimize
a = tf.Variable(0.1) # Learing rate, alpha
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

#Before starting, initialize the variables. We will run this first.
#init = tf.initialize_all_variables()
init = tf.global_variables_initializer()

#Launch the graph.
sess = tf.Session()
sess.run(init)


for step in range(2001):
	sess.run(train)
	if step % 20 == 0:
		print(step, sess.run(cost), sess.run(W), sess.run(b))
