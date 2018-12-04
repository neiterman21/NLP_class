#!/usr/local/bin python

import numpy as np # linear algebra
import tensorflow as tf
import read_data as rd
import sklearn
from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt
from  Datakeeper import *




#tf.set_random_seed(2)
#np.random.seed(2)


def main():
    # Parameters
    learning_rate = 0.01
    training_epochs =2
    batch_size = 20


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

    data_image_train, data_image_test , data_label_train , data_label_test = train_test_split(image_list , numeric_labels , test_size=0.20)

    train_data  = DataKeeper(data_image_train,data_label_train, label_names )
    train_data.setBatchSize(batch_size)
    test_data   = DataKeeper(data_image_test,data_label_test, label_names )


    # Set model weights
    W = tf.Variable(tf.zeros([IMG_HEIGHT*IMG_WIDTH*3, len(label_names)]) , name='weights')
    b = tf.Variable(tf.zeros([len(label_names)]) , name='bias')

    # tf Graph Input
    x = tf.placeholder(tf.float32, [None, IMG_HEIGHT*IMG_WIDTH*3])
    y = tf.placeholder(tf.float32, [None, len(label_names)])
    m = tf.Variable(0.)

    #Construct model
    pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax
    # Minimize error using cross entropy
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = pred, labels = y))
    #cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
    cost_history = np.empty(shape=[1],dtype=float)
    # Gradient Descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()
    msum = tf.summary.scalar('m', m)
    losssum = tf.summary.scalar('loss', cost)
    merged = tf.summary.merge_all()


    # Start training
    with tf.Session() as sess:
        # Run the initializer
        sess.run(init)

        file_writer = tf.summary.FileWriter('./my_graph', sess.graph)
        print("initial cost: " + str(cost))

        # Training cycle
        for epoch in range(training_epochs):

            for i in range(train_data.getNumOfBatches()):
                images , labels = train_data.getNextBatch()
                _, c , curr_summery= sess.run([optimizer,cost, merged], feed_dict={x: images, y: labels})
                if not i % 10:
                    print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c))
                cost_history = np.append(cost_history, c)
                file_writer.add_summary(curr_summery,epoch)

        print("Optimization Finished!")
        # Calculate the correct predictions

        correct_prediction = tf.to_float(tf.greater(pred, 0.5))

        # Calculate accuracy on the test set

        accuracy = tf.reduce_mean(tf.to_float(tf.equal(y, correct_prediction)))
        images , labels = test_data.getNextBatch()
        test_acc = sess.run(accuracy, feed_dict={x: images, y: labels})
        print("test_acc=: {:5f}".format(test_acc) )

        file_writer.close()
        print(sess.run(m))



#
#    # loss function
#    plt.plot(cost)
#    plt.title('Cross Entropy Loss')
#    plt.xlabel('epoch')
#    plt.ylabel('loss')
#    plt.show()


if __name__ == "__main__":
    main()
