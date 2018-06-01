import os
import tensorflow as tf

# Suppress warnings and log information.
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# # Graph definition.
# a = tf.constant(2, name='a')
# b = tf.constant(3, name='b')
# x = tf.add(a,b, name='add')

# # Create summary writer.
# writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())

# # Run the graph.
# with tf.Session() as sess:
# 	print(sess.run(x))

# # Close the writer.
# writer.close()

# ### Constants, Sequences, Variables, Ops
# a = tf.constant([2,2], name='a')
# b = tf.constant([[0,1], [2,3]], name='b')
# x = tf.multiply(a,b, name='mul')

# with tf.Session() as sess:
# 	print(sess.run(x))

# asdf = tf.zeros([2,3], tf.int32)
# asdf2 = tf.zeros_like(asdf)

# # likewise ... ones...ones_like

# asdf = tf.fill([3,3], 9)
# asdf2 = tf.linspace(1., 10., 10)
# asdf3 = tf.range(1., 10., 2.)
# with tf.Session() as sess:
# 	print(sess.run(asdf))
# 	print(sess.run(asdf2))
# 	print(sess.run(asdf3))

# my_const = tf.constant([1.0, 2.0], name="my_const")
# with tf.Session() as sess:
# 	print(sess.graph.as_graph_def())

# W = tf.Variable(tf.truncated_normal([700, 10]))
# with tf.Session() as sess:
# 	sess.run(W.initializer)
# 	print(W.eval())

# x = tf.Variable(10, name='x')
# y = tf.Variable(20, name='y')
# z = tf.add(x, y, name='add')

# writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())
# with tf.Session() as sess:
# 	sess.run(tf.global_variables_initializer())
# 	for _ in range(10):
# 		sess.run(tf.add(x,y,name='add')) # someone decides to be clever to save one line of code
# 	print(sess.graph.as_graph_def())
# writer.close()

a = tf.placeholder(tf.float32, shape=[3]) # a is placeholder for a vector of 3 elements
b = tf.constant([5, 5, 5], tf.float32)
c = a + b # use the placeholder as you would any tensor
writer = tf.summary.FileWriter('graphs/placeholders', tf.get_default_graph())
with tf.Session() as sess:
	print(sess.run(c, {a: [1, 2, 3]})) 
writer.close()


