# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 22:13:02 2019

@author: lenovo
"""

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#设置超参数
learning_rate = 0.00001
train_epochs = 30
batch_size = 100
display_step = 5
#reg = 0.01/(784*256) #for regulization

#设置网络参数
n_hidden1 = 256
n_hidden2 = 256
n_classes = 10
input_dim = 784

#定义占位符
x = tf.placeholder(tf.float32, [None, input_dim])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)
#设置权重变量
weights = {
        "w1": tf.Variable(tf.truncated_normal([input_dim, n_hidden1])*0.001),
        "w2": tf.Variable(tf.truncated_normal([n_hidden1, n_hidden2])*0.001),
        "w_out": tf.Variable(tf.truncated_normal([n_hidden2, n_classes])*0.001)
}

biases = {
        "b1": tf.Variable(tf.zeros([n_hidden1])),
        "b2": tf.Variable(tf.zeros([n_hidden2])),
        "b_out": tf.Variable(tf.zeros([n_classes])) 
}

#creat graph
def multilayer_build(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['w1']), biases['b1'])
    #先dropout再激活
    layer_1 = tf.nn.dropout(layer_1, keep_prob)
    layer_1 = tf.nn.leaky_relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['w2']), biases['b2'])
    layer_2 = tf.nn.dropout(layer_2, keep_prob)
    layer_2 = tf.nn.leaky_relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['w_out']) + biases['b_out']
    return out_layer

pred = multilayer_build(x, weights, biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred,
                                                              labels = y))
#cost += (tf.nn.l2_loss(weights['w1'])*reg+tf.nn.l2_loss(weights['w2'])*reg)
train = tf.train.AdamOptimizer(learning_rate).minimize(cost)

#启动准备好保存
saver = tf.train.Saver()
model_path = "log/521model.ckpt"

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(train_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples/batch_size)
    for i in range(total_batch):
        x1, y1 = mnist.train.next_batch(batch_size)
        _, loss = sess.run([train, cost], feed_dict = {
                                                    x: x1,
                                                    y: y1,
                                                    keep_prob: 0.8
                                                    })
        avg_cost += loss/total_batch
    if epoch % display_step == 0:
        print("Steps at {:.1f}".format(epoch),"cost = {:.9f}".format(loss))

print("finished!")

# 测试 model
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
#注意上面没用run，并没用真正求值，相当于建立计算图
# 计算准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print ("Accuracy:{:.2%}".format(sess.run(accuracy, feed_dict = {
                                                    x: mnist.test.images,
                                                    y: mnist.test.labels,
                                                    keep_prob: 1
                                                    })))

saver.save(sess, model_path)
print("Already saved in ", model_path)
sess.close()








