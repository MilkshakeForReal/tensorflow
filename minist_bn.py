# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 15:44:51 2019

@author: lenovo
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data_sparse/")
import numpy as np


tf.reset_default_graph()

#设置超参数
n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10
learning_rate = 0.001
n_epochs = 50
batch_size = 200

#构建计算图
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")
training = tf.placeholder_with_default(False, shape=(), name='training')

hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1", use_bias = None)
bn1 = tf.layers.batch_normalization(hidden1, training=training, momentum=0.9)
bn1_act = tf.nn.elu(bn1)

hidden2 = tf.layers.dense(bn1_act, n_hidden2, name="hidden2", use_bias = None)
bn2 = tf.layers.batch_normalization(hidden2, training=training, momentum=0.9)
bn2_act = tf.nn.elu(bn2)

logits_before_bn = tf.layers.dense(bn2_act, n_outputs, name="outputs",
                                   use_bias = None)
logits = tf.layers.batch_normalization(logits_before_bn, training=training,
                                      momentum=0.9)

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=logits)#labels允许的数据类型有int32, int64
    loss = tf.reduce_mean(xentropy,name="loss")
with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)
    
"""or you can write this
with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        train_op = optimizer.minimize(loss)
"""


with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits,y,1) #取值最高的一位
    accuracy = tf.reduce_mean(tf.cast(correct,tf.float32)) #结果boolean转为0,1
    

init = tf.global_variables_initializer()
saver = tf.train.Saver()

#tf.GraphKeys.UPDATE_OPS，这是一个tensorflow的计算图中内置的一个集合
extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)


with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run([training_op, extra_update_ops],
                    feed_dict={training: True, X: X_batch, y: y_batch})
        accuracy_val = accuracy.eval(feed_dict={X: mnist.test.images,
                                                y: mnist.test.labels})
        print(epoch, "Test accuracy:{:.2%}".format(accuracy_val))
