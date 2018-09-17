#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : Kerr
@License : Copyright(C) 2018, Get
@Contact : 905392619@qq.com
@Software: PyCharm
@File    : mnist2.py
@Time    : 2018-09-17 14:15
@Desc    : 梯度下降 softmox回归第2版
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

INPUT_NODE = 784
OUTPUT_NODE = 10

# 梯度下降率
GDDOWN = 0.01

# 训练轮数
TRAINING_TIMES = 1000

# 每次训练数量
TRAINING_STEPS = 100

if __name__ == '__main__':
    mnist = input_data.read_data_sets("./MNIST_data", one_hot=True)
    ##首先打印出mnist图片数据的大小
    print("input_data\'s train size: ", mnist.train.num_examples)
    print("input_data\'s validation size: ", mnist.validation.num_examples)
    print("input_data\'s test size: ", mnist.test.num_examples)

    # 各个变量
    x = tf.placeholder("float", shape=[None, INPUT_NODE])
    y_ = tf.placeholder("float", shape=[None, OUTPUT_NODE])

    W = tf.Variable(tf.zeros([INPUT_NODE, OUTPUT_NODE]))
    b = tf.Variable(tf.zeros([OUTPUT_NODE]))

    # 变量初始化
    init = tf.global_variables_initializer()
    sess = tf.InteractiveSession()
    sess.run(init)

    # 初始化图
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    # 优化算法
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    train_step = tf.train.GradientDescentOptimizer(GDDOWN).minimize(cross_entropy)

    for i in range(TRAINING_TIMES):
        # 训练
        batch_xs, batch_ys = mnist.train.next_batch(TRAINING_STEPS)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        # 模型评估
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        j = i + 1
        print("第%d轮训练,训练个数%d个" % (j, j * TRAINING_STEPS))
        # print("正确率预测： " + correct_prediction + "\n")
        print("当前正确率： ")
        print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
