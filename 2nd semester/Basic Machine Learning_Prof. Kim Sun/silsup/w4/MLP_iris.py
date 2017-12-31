#################
### MLP for custom data
#################
import numpy as np
import tensorflow as tf
from sklearn import datasets, model_selection
import math


### define hyperparameters
n_classes = 3
n_features = 4
n_hidden_1 = 4
n_hidden_2 = 4

learning_rate = 0.01
max_iter = 100000

### define graph
x = tf.placeholder(tf.float32, shape=[None, n_features], name="X")
y_label = tf.placeholder(tf.int32, shape=[None], name="Y_label")
# one-hot encoding
y = tf.one_hot(indices=y_label, depth=n_classes)

# hidden 1
with tf.name_scope('hidden1'):
  weights1 = tf.Variable(
      tf.truncated_normal([n_features, n_hidden_1],
                          stddev=1.0),
      name='weights')
  biases1 = tf.Variable(
      tf.zeros([n_hidden_1]),
      name='biases')
  hidden1 = tf.nn.relu(tf.matmul(x, weights1) + biases1)
# hidden 2
with tf.name_scope('hidden2'):
  weights2 = tf.Variable(
      tf.truncated_normal([n_hidden_1, n_hidden_2],
                          stddev=1.0),
      name='weights')
  biases2 = tf.Variable(
      tf.zeros([n_hidden_2]),
      name='biases')
  hidden2 = tf.nn.relu(tf.matmul(hidden1, weights2) + biases2)
# Linear
with tf.name_scope('softmax_linear'):
  weights3 = tf.Variable(
      tf.truncated_normal([n_hidden_2, n_classes],
                          stddev=1.0),
      name='weights')
  biases3 = tf.Variable(
      tf.zeros([n_classes]),
      name='biases')
  logits = tf.matmul(hidden2, weights3) + biases3

pred = tf.cast( tf.argmax(logits, 1), tf.int32 )
accuracy = tf.reduce_mean( tf.cast( tf.equal(pred, y_label), tf.float32 ))

### define loss
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)





### define input data
# Load the iris dataset
iris = datasets.load_iris()

# iris has two attributes: data, target
print(iris.data.shape)
print(iris.target.shape)

# Split the data into training/testing sets
x_train, x_test, y_train, y_test = model_selection.train_test_split(
  iris.data, iris.target, test_size=0.3)



### Starting sessions
with tf.Session() as sess:
  ## initialize variables
  init = tf.global_variables_initializer()
  sess.run(init)

  ## training
  for i in range(max_iter):
    _, accuracy_val, pred_val, loss_val = sess.run(
      [train_step, accuracy, pred, loss], 
      feed_dict={x: x_train, y_label: y_train})

    if i % 1000 == 0:
      print('=========== Epoch: %d ===========' % i)
      print('Loss', loss_val)
      print('Accuracy', accuracy_val)
      print('Y_prediction ', pred_val[:10])
      print('True', y_train[:10])
      
      # accuracy for testset
      test_accuracy, test_pred = sess.run( 
        [accuracy, pred],
        feed_dict={x: x_test, y_label: y_test})
      print('---- evaluation ----')
      print('acc: %.4f' % test_accuracy)
      print('pred', test_pred)
      print('true', y_test)


    if math.isnan(loss_val):
      print('LOSS is NAN!')
      break