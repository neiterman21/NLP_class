#!/usr/local/bin python

import numpy as np # linear algebra
import tensorflow as tf
import read_data as rd
import sklearn
from sklearn.model_selection import train_test_split
import cv2



# Image Parameters
N_CLASSES = 2 # CHANGE HERE, total number of classes
IMG_WIDTH = 535 # CHANGE HERE, the image height to be resized to
IMG_HEIGHT= 396 # CHANGE HERE, the image width to be resized to
CHANNELS = 3 # The 3 color channels, change to 1 if grayscale

tf.set_random_seed(0)
np.random.seed(0)




def parse_function(filename , label):
    image_string = tf.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize_images(image, (IMG_HEIGHT, IMG_WIDTH))
    image = tf.reshape(image , [IMG_HEIGHT * IMG_WIDTH *3 ,])
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label

def train_preprocess(image, label):
#    image = tf.image.random_flip_left_right(image)

 #   image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
 #   image = tf.image.random_saturation(image, lower=0.5, upper=1.5)

    # Make sure the image is still in [0, 1]
  #  image = tf.clip_by_value(image, 0.0, 1.0)

    return image, label

def read_np_images(imagepath , lable):
    imTr = []
    lTr  = []
    for path , l in zip(imagepath , lable) :
        im = cv2.imread(path )
        imTr.append(cv2.resize(im, (IMG_WIDTH,IMG_HEIGHT), interpolation=cv2.INTER_CUBIC))
        lTr.append(np.float32(l))
    return   imTr ,  lTr

def main():
    # Parameters
    learning_rate = 0.01
    training_epochs =100
    batch_size = 100
    display_step = 1


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


    imTr ,lTr = read_np_images(data_image_train , data_label_train)
    imTr = np.array(imTr, dtype='float32') #as mnist
    imTr = np.reshape(imTr,[imTr.shape[0],imTr.shape[1]*imTr.shape[2]*imTr.shape[3]])
    a = np.zeros((len(data_label_train),len(label_names)))
    for line , i in  zip(a, range(len(data_label_train))):
        line[data_label_train[i]] = 1
    lTr = a
    lTr = np.array(lTr,dtype='float64')

    imTe,lTe = read_np_images(data_image_test , data_label_test)
    imTe = np.array(imTe,dtype='float32') #as mnist
    imTe = np.reshape(imTe,[imTe.shape[0],imTe.shape[1]*imTe.shape[2]*imTe.shape[3]])

    a = np.zeros((len(data_label_test),len(label_names)))
    for line , i in  zip(a, range(len(data_label_test))):
        line[data_label_test[i]] = 1
    lTe = a
    lTe = np.array(lTr,dtype='float64')

    # Set model weights
    W = tf.Variable(tf.zeros([IMG_HEIGHT*IMG_WIDTH*3, len(label_names)]) , name='weights')
    b = tf.Variable(tf.zeros([len(label_names)]) , name='bias')

    # tf Graph Input
    x = tf.placeholder(tf.float32, [None, IMG_HEIGHT*IMG_WIDTH*3])
    y = tf.placeholder(tf.float32, [None, len(label_names)])


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

    # Start training
    with tf.Session() as sess:

        # Run the initializer
        sess.run(init)

        # Training cycle
        for epoch in range(training_epochs):

            for start, end in zip(range(0, len(imTr), 128), range(128, len(imTr)+1, 128)):

                _, c = sess.run([optimizer,cost], feed_dict={x: imTr[start:end], y: lTr[start:end]})
                print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c))
                cost_history = np.append(cost_history, c)

        # Calculate the correct predictions

        correct_prediction = tf.to_float(tf.greater(pred, 0.5))

        # Calculate accuracy on the test set

        accuracy = tf.reduce_mean(tf.to_float(tf.equal(y, correct_prediction)))
        print("accurasy: ", '{:.9f}'.format(accuracy))
        print(i, np.mean(np.argmax(lTe, axis=1) == sess.run(pred, feed_dict={x: imTe})))


        print("Optimization Finished!")


if __name__ == "__main__":
    main()
