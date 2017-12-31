from tensorflow import flags
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./data", one_hot=True)
FLAGS = flags.FLAGS

flags.DEFINE_integer("batch_size", 128, "number of batch size. default 128.")
flags.DEFINE_float("learning_rate", 0.01, "initial learning rate.")
flags.DEFINE_integer("max_steps", 10000, "max steps to train.")

batch_size = FLAGS.batch_size
learning_rate = FLAGS.learning_rate
max_step = FLAGS.max_steps



model_inputs = tf.placeholder(dtype = tf.float32, shape= [None, 784])
labels = tf.placeholder(dtype=tf.float32, shape=[None, 10])

w = tf.Variable(tf.random_normal(shape=[784, 10]))
b = tf.Variable(tf.random_normal(shape=[10]))

logits = tf.matmul(model_inputs, w) + b
predictions = tf.nn.softmax(logits)

loss = tf.losses.softmax_cross_entropy(
    onehot_labels=labels,
    logits = predictions)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
train_op = optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(1000):
        batch_images, batch_labels = mnist.train.next_batch(100)
        feed = {model_inputs: batch_images, labels: batch_labels}
        _, loss_val = sess.run([train_op, loss], feed_dict=feed)
        print "step{}|loss:{}". format(step, loss_val)
