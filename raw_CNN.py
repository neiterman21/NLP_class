from __future__ import division, print_function, absolute_import

import tensorflow as tf
from  Datakeeper import *
from sklearn.model_selection import train_test_split
import read_data as rd


# Training Parameters
learning_rate = 0.0001
num_steps = 200
batch_size = 128
display_step = 5
training_epochs = 10

# Network Parameters
num_input = IMG_HEIGHT*IMG_WIDTH*3
num_classes = 6
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
X = tf.placeholder(tf.float32, [None, num_input])
Y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=10):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    x = tf.reshape(x, shape=[-1, IMG_HEIGHT, IMG_WIDTH, 3])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    print(weights['wd1'].get_shape())
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    print(tf.shape(out))
    return out

# Store layers weight & bias
weights = {
    # 32x32 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([IMG_HEIGHT, 20 , 3, 32])),
    # 32x32 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([16, 16, 32, 64])),
    # fully connected, 8*8*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([16*16*64, 1024])),
    # 1024 inputs, 6 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, num_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
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
    # Construct model cnn
    logits = conv_net(X, weights, biases, keep_prob)
    prediction = tf.nn.softmax(logits)
    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

    # Evaluate model
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
                # Run optimization op (backprop)
                sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.8})
                if epoc % display_step == 0 or epoc == 1:
                    # Calculate batch loss and accuracy
                    loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                         Y: batch_y,
                                                                         keep_prob: 1.0})
                    print("epoc " + str(epoc) + ", Minibatch Loss= " + \
                          "{:.4f}".format(loss) + ", Training Accuracy= " + \
                          "{:.3f}".format(acc))

        print("Optimization Finished!")

        test_x, test_y = test_data.getNextBatch()
        print("Testing Accuracy:", \
            sess.run(accuracy, feed_dict={X: test_x,
                                          Y: test_y,
                                          keep_prob: 1.0}))

        return
'''
    # Start training
    with tf.Session() as sess:
        sess.run(init)
        # Test model
        pred = tf.nn.softmax(logits)  # Apply softmax to logits
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        # Training cycle
        for epoch in range(training_epochs):
            total_batch = train_data.getNumOfBatches()
            # Loop over all batches
            for i in range(total_batch):
                batch_x, batch_y = train_data.getNextBatch()
                sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, keep_prob: 1.0})


            batch_x_, batch_y_ = train_data.getNextBatch()
            test_x, test_y = test_data.getNextBatch()
            print("Epoc:",epoch, "Train Accuracy:", accuracy.eval({X: batch_x_, Y: batch_y_}) , "Test Accuracy:", accuracy.eval({X: test_x, Y: test_y}))

        print("Optimization Finished!")
'''

if __name__ == "__main__":
    main()
