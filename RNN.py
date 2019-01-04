from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
from  Datakeeper import *
from sklearn.model_selection import train_test_split
import read_data as rd

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

'''
To classify images using a recurrent neural network, we consider every image
row as a sequence of pixels. Because MNIST image shape is 28*28px, we will then
handle 28 sequences of 28 steps for every sample.
'''

# Training Parameters
learning_rate = 0.001
training_steps = 10000
batch_size = 128
display_step = 5

# Network Parameters
num_input = IMG_HEIGHT # MNIST data input (img shape: 28*28)
timesteps = IMG_WIDTH # timesteps
num_hidden = 256 # hidden layer num of features
num_classes = 6 # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, num_classes])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}


def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, timesteps, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

def main():

    training_epochs = 500

    image_list_ , label_list_ = rd.read_labeld_image_list()

    image_list = []
    label_list = []

    for path , lable in zip(image_list_ , label_list_) :
        if "english" in lable or "spanish" in lable or "arabic" in lable or "mandarin" in lable or "french" in lable or "russian" in lable:
            image_list.append(path)
            label_list.append(lable)
    label_names = list(set(label_list))

    numeric_labels = []
    for l in label_list :
        numeric_labels.append(label_names.index(l))

    data_image_train, data_image_test , data_label_train , data_label_test = train_test_split(image_list , numeric_labels , test_size=0.15)

    train_data  = DataKeeper(data_image_train,data_label_train, label_names )
    train_data.setBatchSize(batch_size)
    test_data   = DataKeeper(data_image_test,data_label_test, label_names )
    test_data.setBatchSize(batch_size)
    logits = RNN(X, weights, biases)
    prediction = tf.nn.softmax(logits)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=Y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

    # Evaluate model (with test logits, for dropout to be disabled)
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    # Start training
    with tf.Session() as sess:

        # Run the initializer
        sess.run(init)
        for epoc in range(1, training_epochs+1):
            total_batch = train_data.getNumOfBatches()
            # Loop over all batches
            for i in range(total_batch):
                batch_x, batch_y = train_data.getNextBatch()

                # Reshape data to get 28 seq of 28 elements
                batch_x = batch_x.reshape((batch_size, timesteps, num_input))
                # Run optimization op (backprop)
                sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
                if epoc % display_step == 0 or epoc == 1:
                    # Calculate batch loss and accuracy
                    loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                         Y: batch_y})
                    print("epoc " + str(epoc) + ", Minibatch Loss= " + \
                          "{:.4f}".format(loss) + ", Training Accuracy= " + \
                          "{:.3f}".format(acc))

        print("Optimization Finished!")

        # Calculate accuracy for 128 mnist test images
        test_len = 128
        test_x, test_y = test_data.getNextBatch()
        test_x = test_x.reshape((-1, timesteps, num_input))

        print("Testing Accuracy:", \
            sess.run(accuracy, feed_dict={X: test_x, Y: test_y}))

if __name__ == "__main__":
    main()

