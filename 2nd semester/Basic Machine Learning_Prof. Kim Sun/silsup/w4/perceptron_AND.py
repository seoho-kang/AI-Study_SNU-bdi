#################
### Perceptron for AND function
#################
import tensorflow as tf
import math


### define graph
x = tf.placeholder(tf.float32, shape=(None, 2), name = 'x-input')
y = tf.placeholder(tf.float32, shape=(None, 1), name = 'y-input')

v_weight = tf.Variable(
  tf.random_uniform(shape=(2, 1), minval=-1, maxval=1), 
  dtype=tf.float32, 
  name = "W")
v_bias = tf.Variable(
  tf.zeros(shape=(1)), 
  dtype=tf.float32, 
  name = "w0")

y_h = tf.sigmoid( tf.matmul(x, v_weight) + v_bias )


### define loss function
# # prevent nan loss
# epsilon = 1e-10 
# loss = tf.reduce_mean( 
#   -1 * y * tf.log(y_h + epsilon) - 
#   (1 - y) * tf.log(1 - y_h + epsilon) )

loss = tf.reduce_mean( 
  -1 * y * tf.log(y_h) - 
  (1 - y) * tf.log(1 - y_h) )

### define optimization function
learning_rate = 0.1
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)



### define input data
DATA = {
  'X': [[0,0],[0,1],[1,0],[1,1]],
  'Y': [[0],[1],[1],[1]]
}


### Starting sessions
with tf.Session() as sess:
  ## initialize variables
  init = tf.global_variables_initializer()
  sess.run(init)

  max_iter = 100000

  for i in range(max_iter):
    _, v_w_val, v_b_val, y_h_val, loss_val = sess.run(
      [train_step, v_weight, v_bias, y_h, loss], 
      feed_dict={x: DATA['X'], y: DATA['Y']})
    
    if i % 1000 == 0:
      print('Epoch ', i)
      print('Y_prediction ', y_h_val)
      print('True', DATA['Y'])
      print('Loss', loss_val)
      print('Weight', v_w_val)
      print('Bias', v_b_val)

    if math.isnan(loss_val):
      print('LOSS is NAN!')
      break