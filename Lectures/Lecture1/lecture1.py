import tensorflow as tf

a = tf.add(3,5)
with tf.Session() as sess:
	print(sess.run(a))

x = 2
y = 3
add_op = tf.add(x, y)
mul_op = tf.multiply(x, y)
useless = tf.multiply(x, add_op)
pow_op = tf.pow(add_op, mul_op)
with tf.Session() as sess:
	z = sess.run(pow_op)

# Creates a graph.
with tf.device('/gpu:0'):
  a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], name='a')
  b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], name='b')
  c = tf.multiply(a, b)

# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

# Runs the op.
print(sess.run(c))

g = tf.Graph()
with g.as_default():
	x = tf.add(3, 5)
with tf.Session(graph=g) as sess:
	print(sess.run(x))


g1 = tf.get_default_graph()
g2 = tf.Graph()
# add ops to the default graph
with g1.as_default():
	a = tf.Constant(3)
# add ops to the user created graph
with g2.as_default():
	b = tf.Constant(5)