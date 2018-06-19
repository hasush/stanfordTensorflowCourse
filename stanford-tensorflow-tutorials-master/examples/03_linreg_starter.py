""" Starter code for simple linear regression example using placeholders
Created by Chip Huyen (huyenn@cs.stanford.edu)
CS20: "TensorFlow for Deep Learning Research"
cs20.stanford.edu
Lecture 03
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf

import utils

DATA_FILE = 'data/birth_life_2010.txt'

from tensorflow.examples.tutorials.mnist import input_data


### LINEAR REGRESSION WITH SQUARED LOSS ###

# # Step 1: read in data from the .txt file
# data, n_samples = utils.read_birth_life_data(DATA_FILE)

# print(type(data))
# print(data.shape)

# # Step 2: create placeholders for X (birth rate) and Y (life expectancy)
# # Remember both X and Y are scalars with type float
# X = tf.placeholder(dtype=tf.float32, shape=(), name='BirthRate')
# Y = tf.placeholder(dtype=tf.float32, shape=(), name='LifeExpectancy')

# # Step 3: create weight and bias, initialized to 0.0
# # Make sure to use tf.get_variable
# w = tf.get_variable(name='Weight_1', shape=(), dtype=tf.float32, initializer=tf.zeros_initializer())
# u = tf.get_variable(name='Weight_2', shape=(), dtype=tf.float32, initializer=tf.zeros_initializer())
# b = tf.get_variable(name='Bias', shape=(), dtype=tf.float32, initializer=tf.zeros_initializer())

# # Step 4: build model to predict Y
# # e.g. how would you derive at Y_predicted given X, w, and b
# Y_predicted = w*X + b

# # Step 5: use the square error as the loss function
# loss = tf.square(Y-Y_predicted, name='loss')

# # Step 6: using gradient descent with learning rate of 0.001 to minimize loss
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

# start = time.time()

# # Create a filewriter to write the model's graph to TensorBoard
# writer = tf.summary.FileWriter(logdir="graphs/03_linreg_starter", graph=tf.get_default_graph())

# with tf.Session() as sess:
#     # Step 7: initialize the necessary variables, in this case, w and b
#     sess.run(tf.global_variables_initializer())

#     # Step 8: train the model for 100 epochs
#     for i in range(100):
#         total_loss = 0
#         for x, y in data:
#             # Execute train_op and get the value of loss.
#             # Don't forget to feed in data for placeholders
#             _, loss_out = sess.run([optimizer, loss], feed_dict={X:x, Y:y})
#             total_loss += loss_out

#         print('Epoch {0}: {1}'.format(i, total_loss/n_samples))

#     # close the writer when you're done using it

#     writer.close()
    
#     # Step 9: output the values of w and b
#     w_out, b_out = w.eval(), b.eval()

# print('Took: %f seconds' %(time.time() - start))

# # uncomment the following lines to see the plot 
# plt.plot(data[:,0], data[:,1], 'bo', label='Real data')
# plt.plot(data[:,0], data[:,0] * w_out + b_out, 'rx', label='Predicted data')
# plt.legend()
# plt.show()


### QUADRATIC REGRESSION WITH SQUARE LOSS ###

# # Step 1: read in data from the .txt file
# data, n_samples = utils.read_birth_life_data(DATA_FILE)

# print(type(data))
# print(data.shape)

# # Step 2: create placeholders for X (birth rate) and Y (life expectancy)
# # Remember both X and Y are scalars with type float
# X = tf.placeholder(dtype=tf.float32, shape=(), name='BirthRate')
# Y = tf.placeholder(dtype=tf.float32, shape=(), name='LifeExpectancy')

# # Step 3: create weight and bias, initialized to 0.0
# # Make sure to use tf.get_variable
# w = tf.get_variable(name='Weight_1', shape=(), dtype=tf.float32, initializer=tf.zeros_initializer())
# u = tf.get_variable(name='Weight_2', shape=(), dtype=tf.float32, initializer=tf.zeros_initializer())
# b = tf.get_variable(name='Bias', shape=(), dtype=tf.float32, initializer=tf.zeros_initializer())

# # Step 4: build model to predict Y
# # e.g. how would you derive at Y_predicted given X, w, and b
# Y_predicted = w*X*X + u*X + b

# # Step 5: use the square error as the loss function
# loss = tf.square(Y-Y_predicted, name='loss')

# # Step 6: using gradient descent with learning rate of 0.001 to minimize loss
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

# start = time.time()

# # Create a filewriter to write the model's graph to TensorBoard
# writer = tf.summary.FileWriter(logdir="graphs/03_linreg_starter", graph=tf.get_default_graph())

# with tf.Session() as sess:
#     # Step 7: initialize the necessary variables, in this case, w and b
#     sess.run(tf.global_variables_initializer())

#     # Step 8: train the model for 100 epochs
#     for i in range(100):
#         total_loss = 0
#         for x, y in data:
#             # Execute train_op and get the value of loss.
#             # Don't forget to feed in data for placeholders
#             _, loss_out = sess.run([optimizer, loss], feed_dict={X:x, Y:y})
#             total_loss += loss_out

#         print('Epoch {0}: {1}'.format(i, total_loss/n_samples))

#     # close the writer when you're done using it

#     writer.close()
    
#     # Step 9: output the values of w and b
#     w_out, u_out, b_out = w.eval(), u.eval(), b.eval()

# print('Took: %f seconds' %(time.time() - start))

# # uncomment the following lines to see the plot 
# plt.plot(data[:,0], data[:,1], 'bo', label='Real data')
# outValues = data[:,0]*data[:,0]*w_out + data[:,0]*u_out + b_out
# print(outValues.shape)
# plt.plot(data[:,0], outValues[:], 'r', marker="o", label='Predicted data')
# plt.legend()
# plt.show()


### LINEAR REGRESSION WITH HUBER LOSS ###

# # Step 1: read in data from the .txt file
# data, n_samples = utils.read_birth_life_data(DATA_FILE)

# print(type(data))
# print(data.shape)

# # Step 2: create placeholders for X (birth rate) and Y (life expectancy)
# # Remember both X and Y are scalars with type float
# X = tf.placeholder(dtype=tf.float32, shape=(), name='BirthRate')
# Y = tf.placeholder(dtype=tf.float32, shape=(), name='LifeExpectancy')

# # Step 3: create weight and bias, initialized to 0.0
# # Make sure to use tf.get_variable
# w = tf.get_variable(name='Weight_1', shape=(), dtype=tf.float32, initializer=tf.zeros_initializer())
# u = tf.get_variable(name='Weight_2', shape=(), dtype=tf.float32, initializer=tf.zeros_initializer())
# b = tf.get_variable(name='Bias', shape=(), dtype=tf.float32, initializer=tf.zeros_initializer())

# # Step 4: build model to predict Y
# # e.g. how would you derive at Y_predicted given X, w, and b
# Y_predicted = w*X + b

# def huber_loss(labels, predictions, delta=14.0):
#     residual = tf.abs(labels-predictions)
#     def f1(): return 0.5*tf.square(residual)
#     def f2(): return delta*residual - 0.5*tf.square(delta)
#     return tf.cond(residual < delta, f1, f2)

# # Step 5: use the square error as the loss function
# loss = huber_loss(Y, Y_predicted)

# # Step 6: using gradient descent with learning rate of 0.001 to minimize loss
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

# start = time.time()

# # Create a filewriter to write the model's graph to TensorBoard
# writer = tf.summary.FileWriter(logdir="graphs/03_linreg_starter", graph=tf.get_default_graph())

# with tf.Session() as sess:
#     # Step 7: initialize the necessary variables, in this case, w and b
#     sess.run(tf.global_variables_initializer())

#     # Step 8: train the model for 100 epochs
#     for i in range(100):
#         total_loss = 0
#         for x, y in data:
#             # Execute train_op and get the value of loss.
#             # Don't forget to feed in data for placeholders
#             _, loss_out = sess.run([optimizer, loss], feed_dict={X:x, Y:y})
#             total_loss += loss_out

#         print('Epoch {0}: {1}'.format(i, total_loss/n_samples))

#     # close the writer when you're done using it

#     writer.close()
    
#     # Step 9: output the values of w and b
#     w_out, b_out = w.eval(), b.eval()

# print('Took: %f seconds' %(time.time() - start))

# # uncomment the following lines to see the plot 
# plt.plot(data[:,0], data[:,1], 'bo', label='Real data')
# plt.plot(data[:,0], data[:,0] * w_out + b_out, 'r', label='Predicted data')
# plt.legend()
# plt.show()


### LINEAR REGRESSION WITH HUBER LOSS AND TF DATA ITERATOR AND CHANGING LEARNING RATE ###

# # Step 1: read in data from the .txt file
# data, n_samples = utils.read_birth_life_data(DATA_FILE)
# dataset = tf.data.Dataset.from_tensor_slices((data[:,0], data[:,1]))
# print(type(data))
# print(data.shape)

# # # One shot iterator example. Comment/Unceomment
# # iterator = dataset.make_one_shot_iterator()
# # X, Y = iterator.get_next()

# # with tf.Session() as sess:
# #     try:
# #         while True:
# #             print(sess.run([X,Y]))
# #     except tf.errors.OutOfRangeError:
# #         print("Went out of range.")

# iterator = dataset.make_initializable_iterator()
# X,Y = iterator.get_next()

# # # Step 2: create placeholders for X (birth rate) and Y (life expectancy)
# # # Remember both X and Y are scalars with type float
# # X = tf.placeholder(dtype=tf.float32, shape=(), name='BirthRate')
# # Y = tf.placeholder(dtype=tf.float32, shape=(), name='LifeExpectancy')

# # Step 3: create weight and bias, initialized to 0.0
# # Make sure to use tf.get_variable
# w = tf.get_variable(name='Weight_1', shape=(), dtype=tf.float32, initializer=tf.zeros_initializer())
# u = tf.get_variable(name='Weight_2', shape=(), dtype=tf.float32, initializer=tf.zeros_initializer())
# b = tf.get_variable(name='Bias', shape=(), dtype=tf.float32, initializer=tf.zeros_initializer())

# # Step 4: build model to predict Y
# # e.g. how would you derive at Y_predicted given X, w, and b
# Y_predicted = w*X + b

# def huber_loss(labels, predictions, delta=14.0):
#     residual = tf.abs(labels-predictions)
#     def f1(): return 0.5*tf.square(residual)
#     def f2(): return delta*residual - 0.5*tf.square(delta)
#     return tf.cond(residual < delta, f1, f2)

# # Step 5: use the square error as the loss function
# loss = huber_loss(Y, Y_predicted)

# global_step = tf.Variable(0, trainable=False, dtype=tf.int32)
# learning_rate = 0.01*0.99987**tf.cast(global_step, tf.float32)
# increment_step = global_step.assign_add(1)

# # Step 6: using gradient descent with learning rate of 0.001 to minimize loss
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

# start = time.time()

# # Create a filewriter to write the model's graph to TensorBoard
# writer = tf.summary.FileWriter(logdir="graphs/03_linreg_starter", graph=tf.get_default_graph())

# with tf.Session() as sess:
#     # Step 7: initialize the necessary variables, in this case, w and b
#     sess.run(tf.global_variables_initializer())

#     # Step 8: train the model for 100 epochs
#     for i in range(100):
#         sess.run(iterator.initializer)
#         total_loss = 0
#         try:
#             while True:
#                 _, loss_out = sess.run([optimizer, loss])
#                 total_loss += loss_out
#                 sess.run(increment_step)
#                 # print("{} -- {}", global_step.eval(), learning_rate.eval())
#         except tf.errors.OutOfRangeError:
#             pass

#         print('Epoch {0}: {1}'.format(i, total_loss/n_samples))

#     # close the writer when you're done using it

#     writer.close()
    
#     # Step 9: output the values of w and b
#     w_out, b_out = w.eval(), b.eval()

# print('Took: %f seconds' %(time.time() - start))

# # uncomment the following lines to see the plot 
# plt.plot(data[:,0], data[:,1], 'bo', label='Real data')
# plt.plot(data[:,0], data[:,0] * w_out + b_out, 'r', label='Predicted data')
# plt.legend()
# plt.show()

### LOGISTIC REGRESION WITH MNIST ###
batch_size = 128
num_epochs = 30
learning_rate = 0.01
num_test = 10000
num_val = 5000

# Download mnist data.
mnist_folder = 'data/mnist'
utils.download_mnist(mnist_folder)

# Split into different data sets.
train, val, test = utils.read_mnist(mnist_folder, flatten=True)

# Create tensorfor dataset objects. Train, val, and test are already tuples of (features, labels) numpy arrays.
train_data = tf.data.Dataset.from_tensor_slices(train)
train_data = train_data.shuffle(10000)
test_data = tf.data.Dataset.from_tensor_slices(test)
val_data = tf.data.Dataset.from_tensor_slices(val)

# Seperate into batches.
train_data = train_data.batch(batch_size)
test_data = test_data.batch(batch_size)
val_data = val_data.batch(batch_size)

# Create iterator and initializers.
iterator = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)
img, label = iterator.get_next()
train_init = iterator.make_initializer(train_data)
test_init = iterator.make_initializer(test_data)
val_init = iterator.make_initializer(val_data)


print(train[0][0].shape)

# Variables for network architecture.
w = tf.get_variable(dtype=tf.float32, name='weights', shape=(784, 10), initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
b = tf.get_variable(dtype=tf.float32, name='bias', shape=(1, 10), initializer=tf.zeros_initializer())

# Compute the network output.
logits = tf.matmul(img, w) + b

# Compute the entropy of the prediction.
entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label, name='entropy')

# Loss given entropy.
loss = tf.reduce_sum(entropy, name='loss')

# Optimizer for training.
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Prediction.
prediction = tf.nn.softmax(logits)
correctPrediction = tf.equal(tf.argmax(prediction,1), tf.argmax(label,1))
accuracy = tf.reduce_sum(tf.cast(correctPrediction, tf.float32))

writer = tf.summary.FileWriter(logdir='graphs/mnistLogisticClassification', graph=tf.get_default_graph())

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(num_epochs):
        sess.run(train_init)
        total_loss = 0
        n_batches = 0
        try:
            while True:
                sess.run(optimizer)
                total_loss += sess.run(loss)
                n_batches +=1
        except tf.errors.OutOfRangeError:
            pass

        sess.run(val_init)
        total_accuracy = 0
        try:
            while True:
                total_accuracy += sess.run(accuracy)
        except tf.errors.OutOfRangeError:
            pass

        print("Epoch: {} -- Loss: {} -- Val Accuracy: {}".format(epoch, total_loss/n_batches, total_accuracy/num_val))


    sess.run(test_init)
    total_accuracy = 0
    try:
        while True:
            total_accuracy += sess.run(accuracy)
    except tf.errors.OutOfRangeError:
        pass

    writer.close()


    print("Accuracy: {}".format(total_accuracy/num_test))
