#!/usr/local/bin python

import numpy as np  # linear algebra
import tensorflow as tf
import read_data as rd
import sklearn
from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt
from Datakeeper import *
from tensorflow.contrib.boosted_trees.lib.learner import batch
from termcolor import colored

# tf.set_random_seed(2)
# np.random.seed(2)

(hidden1_size, hidden2_size, hidden3_size) = (128, 64, 32)
# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.truncated_normal([IMG_HEIGHT * IMG_WIDTH * 3, hidden1_size], stddev=0.1)),
    'h2': tf.Variable(tf.truncated_normal([hidden1_size, hidden2_size], stddev=0.1)),
    #  'h3': tf.Variable(tf.random_normal([hidden2_size, hidden3_size])),
    'out': tf.Variable(tf.truncated_normal([hidden2_size, 6], stddev=0.1))
}

biases = {
    'b1': tf.Variable(tf.constant(0.1, shape=[hidden1_size])),
    'b2': tf.Variable(tf.constant(0.1, shape=[hidden2_size])),
    # 'b3': tf.Variable(tf.random_normal([hidden3_size])),
    'out': tf.Variable(tf.constant(0.1, shape=[6]))
}


# Create model
def multilayer_perceptron(x):
    # Hidden fully connected layer with 512 neurons
    layer_1 = tf.nn.relu(tf.matmul(x, weights['h1']) + biases['b1'])
    # Hidden fully connected layer with 128 neurons
    layer_2 = tf.nn.relu(tf.matmul(layer_1, weights['h2']) + biases['b2'])
    # Output fully connected layer with a neuron for each class
    # layer_3 = tf.nn.relu(tf.matmul(layer_2, weights['h3']) + biases['b3'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer


def main():
    # Parameters
    learning_rate = 0.0001
    training_epochs = 100
    batch_size = 96
    display_step = 1

    np.set_printoptions(threshold=np.nan)

    image_list, label_list, label_names, numeric_labels = rd.get_image_and_label()

    data_image_train, data_image_test, data_label_train, data_label_test = train_test_split(image_list, numeric_labels,
                                                                                            test_size=0.15)

    train_data = DataKeeper(data_image_train, data_label_train, label_names)
    train_data.setBatchSize(batch_size)
    test_data = DataKeeper(data_image_test, data_label_test, label_names)

    x = tf.placeholder(tf.float32, [None, IMG_HEIGHT * IMG_WIDTH * 3])

    y_ = tf.placeholder(tf.float32, [None, len(label_names)])

    # Construct model
    logits = multilayer_perceptron(x)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)
    # Initializing the variables
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        # Test model
        pred = tf.nn.softmax(logits)  # Apply softmax to logits
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y_, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        # Training cycle
        for epoch in range(training_epochs):
            total_batch = train_data.getNumOfBatches()
            # Loop over all batches
            for i in range(total_batch):
                batch_x, batch_y = train_data.getNextBatch()
                sess.run(train_op, feed_dict={x: batch_x, y_: batch_y})

            batch_x_, batch_y_ = train_data.getNextBatch()
            test_x, test_y = test_data.getNextBatch()
            print("Epoc:", epoch, "Train Accuracy:", accuracy.eval({x: batch_x_, y_: batch_y_}), "Test Accuracy:",
                  accuracy.eval({x: test_x, y_: test_y}))

        print("Optimization Finished!")


if __name__ == "__main__":
    main()
