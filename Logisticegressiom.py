# model definition (canonical way)
import tensorflow as tf
class LogisticRegression(tf.keras.Model):

    def __init__(self, num_classes):
        super(LogisticRegression, self).__init__()
        self.dense = tf.keras.layers.Dense(num_classes)

    def call(self, inputs, training=None, mask=None):
        output = self.dense(inputs)

        # softmax op does not exist on the gpu
        with tf.device('/cpu:0'):
            output = tf.nn.softmax(output)

        return output
