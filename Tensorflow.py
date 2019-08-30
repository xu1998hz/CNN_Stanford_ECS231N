import numpy as np
import tensorflow as tf

# Define computational graph
N, D, H = 64, 1000, 100
## Create placeholders for input x, weights w1 and w2 and targets y
x = tf.placeholder(tf.float32, shape=(N, D))
y = tf.placeholder(tf.float32, shape=(N, D))
# Change w1 and w2 from placeholder to Vairable
#w1 = tf.placeholder(tf.float32, shape=(D, H))
#w2 = tf.placeholder(tf.float32, shape=(H, D))
w1 = tf.Variable(tf.random_normal((D,H)))
w2 = tf.Variable(tf.random_normal((H,D)))

# Forward pass: compute prediction for y and loss
h = tf.maximum(tf.matmul(x, w1), 0)
y_pred = tf.matmul(h, w2)
# diff = y_pred - y
# loss = tf.reduce_mean(tf.reduce_sum(diff**2, axis=1))
# predefined common losses
loss = tf.losses.mean_squared_error(y_pred, y)

# Tell TensorFLow to compute loss of gradient with respect to w1 and w2
# grad_w1, grad_w2 = tf.gradients(loss, [w1, w2])
# Can use an optimizer to compute gradients and update weights
optimizer = tf.train.GradientDescentOptimizer(le-5)
updates = optimizer.minimize(loss)

# Now done building our graph, so we enter a session so we can actually run the graph
with tf.Session() as sess:
	# Initialize weights w1 and w2, global_variable knows placeholders are fed outside, variables 
	# are initialized inside
	sess.run(tf.global_variables_initializer())
	# Create numpy arrays that will fit in the placeholders above
	values = {x: np.random.randn(N, D),
		  y: np.random.randn(N, D)}
	losses = []	
	"""
	# Run the graph: feed in the numpy arrays for x, y, w1 and w2
	# get numpy arrays for loss, grad_w1 and grad_w2
	out = sess.run([loss, grad_w1, grad_w2], 
			feed_dict=values)
	loss_val, grad_w1_val, grad_w2_val = out """
	for t in range(50):
		loss_val, _ = sess.run([loss, updates], feed_dict=values)	

