""" Solution for simple logistic regression model for MNIST
with placeholder
MNIST dataset: yann.lecun.com/exdb/mnist/
Created by Chip Huyen (huyenn@cs.stanford.edu)
CS20: "TensorFlow for Deep Learning Research"
cs20.stanford.edu
Lecture 03
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time
import utils
import sys
import matplotlib.pyplot as plt

# # Define paramaters for the model
learning_rate = 0.01
batch_size = 128
n_epochs = 30

# Load data.
mnist = input_data.read_data_sets('data/mnist', one_hot=True)
# X_batch_train, Y_batch_train = mnist.train.next_batch(batch_size)
# X_batch_validation, Y_batch_validation = mnist.validation.next_batch(batch_size)
# X_batch_test, Y_batch_test = mnist.test.next_batch(batch_size)
# X_validation, Y_validation = mnist.validation._images, mnist.validation._labels
# X_test, Y_test = mnist.test._images, mnist.test._labels

# Create placeholders for the features and labels.
X = tf.placeholder(tf.float32, [batch_size, 784], name='image') 
Y = tf.placeholder(tf.float32, [batch_size, 10], name='label')
dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')

# Convolutional Network
def cnn_model(X, dropout_prob):
	input_layer = tf.reshape(X, [batch_size, 28, 28, 1])
	conv1 = tf.layers.conv2d(
		inputs=input_layer,
		filters=32,
		kernel_size=[5,5],
		padding='same',
		activation=tf.nn.relu)
	pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], strides=2)
	conv2 = tf.layers.conv2d(
		inputs=pool1,
		filters=64,
		kernel_size=[5,5],
		padding='same',
		activation=tf.nn.relu)
	pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=2)
	pool2_flat = tf.reshape(pool2,[batch_size, 7*7*64])
	dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
	dropout = tf.layers.dropout(inputs=dense, rate=dropout_prob)
	logits = tf.layers.dense(inputs=dropout, units=10)
	return logits

# Normal Neural Network
def perceptron_model(X, dropout_prob):
	input_layer = X
	dense1 = tf.layers.dense(inputs=input_layer, units=1024, activation=tf.nn.relu)
	dropout1 = tf.layers.dropout(inputs=dense1, rate=dropout_prob)
	dense2 = tf.layers.dense(inputs=dropout1, units=1024, activation=tf.nn.relu)
	dropout2 = tf.layers.dropout(inputs=dense2, rate=dropout_prob)
	logits = tf.layers.dense(inputs=dropout2, units=10)
	return logits
	
# Create the model.
# logits = cnn_model(X, dropout_prob)
logits = perceptron_model(X, dropout_prob)

# Cost and optimizer.
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer().minimize(cost)

# Accuracy of prediction.
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the writer.
writer = tf.summary.FileWriter('./graphs/q2b_perceptron', tf.get_default_graph())

# Run the session.
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	# Define the number of batches for training, validation, and testing.
	num_batches_training = int(mnist.train.num_examples/batch_size)
	num_batches_validation = int(mnist.validation.num_examples/batch_size)
	num_batches_test = int(mnist.test.num_examples/batch_size)

	# Loop over the epochs.
	for epoch in range(n_epochs):

		# Average loss for all batches for this epoch.
		total_loss = 0

		# Train the model.
		for j in range(num_batches_training):

			# Get the next batch of training data.
			X_batch_train, Y_batch_train = mnist.train.next_batch(batch_size)

			# Run the optimizer.
			sess.run(optimizer, feed_dict={X:X_batch_train, Y:Y_batch_train, dropout_prob:0.4})

			# Obtain the loss.
			loss = sess.run(cost, feed_dict={X:X_batch_train, Y:Y_batch_train, dropout_prob:1.0})
			total_loss += loss

		# Average accuracy for all validation examples.
		total_valid_accuracy = 0

		# Test the validation set.
		for j in range(num_batches_validation):

			# Get the next batch of validation data.
			X_batch_validation, Y_batch_validation = mnist.validation.next_batch(batch_size)

			# Compute the accuracy.
			accuracy_out = sess.run(accuracy, feed_dict={X:X_batch_validation, Y:Y_batch_validation, dropout_prob:1.0})
			total_valid_accuracy += accuracy_out

		total_loss = total_loss/num_batches_training
		total_valid_accuracy = total_valid_accuracy/num_batches_validation		

		print("Epoch: {} -- total loss: {} -- total accuracy: {}".format(epoch, total_loss, total_valid_accuracy))


	# Test the testing set.
	total_test_accuracy = 0
	for j in range(num_batches_test):

		# Get the next batch of test data.
		X_batch_test, Y_batch_test = mnist.test.next_batch(batch_size)

		# Compute the accuracy.
		accuracy_out = sess.run(accuracy, feed_dict={X:X_batch_test, Y:Y_batch_test, dropout_prob:1.0})
		total_test_accuracy += accuracy_out

	total_test_accuracy = total_test_accuracy/num_batches_test
	print("Total accuracy on test set: {}".format(total_test_accuracy))

writer.close()		