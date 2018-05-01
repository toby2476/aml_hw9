#Design CNN with 2 convolutional layers, 2 pooling layers, and 2 fully connected layers with 1024 and 10 neurons

#CITATIONS
#used code from the tensorboard tutorial: https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard
#and the tensorflow mnist tutorial: https://www.tensorflow.org/tutorials/layers

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

#Initialize params

rate = 1e-3
batch_size = 100 
rounds = 2001
dense_layer_neurons = 1024

#Setup placeholders and reshape data

x = tf.placeholder(tf.float32, shape=[None,784],name="x") #Pixels in training image
y = tf.placeholder(tf.float32, shape=[None,10],name="labels")	#Label (one-hot)
x_image = tf.reshape(x,[-1,28,28,1])

#Create network

conv1 = tf.layers.conv2d(
      inputs=x_image,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

pool1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

pool2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
flattened = tf.reshape(pool2, [-1,7*7*64])

fc1 = tf.layers.dense(inputs=flattened, units=dense_layer_neurons, activation=tf.nn.relu)
dropout = tf.layers.dropout(inputs=fc1, rate=0.4)
logits = tf.layers.dense(inputs=dropout, units=10)

#Training

with tf.name_scope("cross_entropy"):
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)) #Cross-entropy for loss function

tf.summary.scalar('cross_entropy',cross_entropy)

with tf.name_scope("train"):
	train_step = tf.train.GradientDescentOptimizer(learning_rate=rate).minimize(cross_entropy) 


with tf.name_scope("accuracy"):
	correct_pred = tf.equal(tf.argmax(logits,1),tf.argmax(y,1)) #compute accuracy
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

tf.summary.scalar('accuracy',accuracy)


#Initialize variables
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

#Load Data:
mnist = tf.contrib.learn.datasets.load_dataset("mnist")
train_data = mnist.train.images
train_labels = np.array(mnist.train.labels, dtype=np.int32)

#Setup Filewriter
writer = tf.summary.FileWriter("./Baseline")
merged_summary = tf.summary.merge_all()	

for i in range(rounds):
	batch = mnist.train.next_batch(batch_size)
	one_hot = np.zeros((batch[1].shape[0],10))
	one_hot[np.arange(batch[1].shape[0]),batch[1]] = 1
	if i%100 == 0:
		s = sess.run(merged_summary, feed_dict={x: batch[0], y: one_hot})
		writer.add_summary(s,i)	
	if i%50 == 0:
		[train_accuracy] = sess.run([accuracy], feed_dict={x: batch[0], y: one_hot})
		print("step %d, training accuracy %g" % (i, train_accuracy))
	sess.run(train_step,feed_dict={x: batch[0], y: one_hot})






