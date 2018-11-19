#!/usr/local/bin python

import numpy as np # linear algebra
import tensorflow as tf
import read_data as rd
from collections import Counter



# Image Parameters
N_CLASSES = 2 # CHANGE HERE, total number of classes
IMG_HEIGHT = 28 #535 # CHANGE HERE, the image height to be resized to
IMG_WIDTH = 28 #396 # CHANGE HERE, the image width to be resized to
CHANNELS = 3 # The 3 color channels, change to 1 if grayscale


tf.enable_eager_execution()
tfe = tf.contrib.eager
tf.set_random_seed(0)
np.random.seed(0)

# Set model weights
W = tfe.Variable(tf.zeros([IMG_HEIGHT*IMG_WIDTH*3, 7]) , name='weights')
b = tfe.Variable(tf.zeros([7]) , name='bias')


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

# Logistic regression (Wx + b)
def logistic_regression(inputs):
    return tf.matmul(inputs, W) + b


# Cross-Entropy loss function
def loss_fn(inference_fn, inputs, labels):
    # Using sparse_softmax cross entropy
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=inference_fn(inputs), labels=labels))


# Calculate accuracy
def accuracy_fn(inference_fn, inputs, labels):
    prediction = tf.nn.softmax(inference_fn(inputs))
    correct_pred = tf.equal(tf.argmax(prediction, 1), labels)
    return tf.reduce_mean(tf.cast(correct_pred, tf.float32))

def main():
    # Parameters
    learning_rate = 0.01
    training_epochs = 22
    batch_size = 128
    display_step = 100


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

    dataset = tf.data.Dataset.from_tensor_slices((image_list ,numeric_labels ))
    dataset = dataset.shuffle(len(image_list))
    dataset = dataset.map(parse_function, num_parallel_calls=4)
    dataset = dataset.map(train_preprocess, num_parallel_calls=4)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(batch_size)
    total_batch = len(image_list)/batch_size

    print(dataset)



    # SGD Optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    # Compute gradients
    grad = tfe.implicit_gradients(loss_fn)

    # Training
    average_loss = 0.
    average_acc = 0.

    for epoch in range(training_epochs):

        dataset_iter = tfe.Iterator(dataset)

        for batch in range(int(total_batch)):

            try:
                # Iterate through the dataset
                d = dataset_iter.next()

                # Images
                x_batch = d[0]
                # Labels
                y_batch = tf.cast(d[1], dtype=tf.int64)

                # Compute the batch loss
                batch_loss = loss_fn(logistic_regression, x_batch, y_batch)
                average_loss += batch_loss
                # Compute the batch accuracy
                batch_accuracy = accuracy_fn(logistic_regression, x_batch, y_batch)
                average_acc += batch_accuracy

                if batch == 0:
                    # Display the initial cost, before optimizing
                    print("Initial loss= {:.9f}".format(average_loss))

                # Update the variables following gradients info
                optimizer.apply_gradients(grad(logistic_regression, x_batch, y_batch))

            except :
                pass

            # Display info
            if (batch + 1) % display_step == 0 or batch == 0:
                if batch > 0:
                    average_loss /= display_step
                    average_acc /= display_step
                print("Step:", '%04d' % (batch + 1), " loss=",
                      "{:.9f}".format(average_loss), " accuracy=",
                      "{:.4f}".format(average_acc))
                average_loss = 0.
                average_acc = 0.


if __name__ == "__main__":
    main()
