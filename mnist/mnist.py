#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : Kerr
@License : Copyright(C) 2018, Get
@Contact : 905392619@qq.com
@Software: PyCharm
@File    : mnist.py
@Time    : 2018-09-013 16:46
@Desc    : mnist手写体识别
"""
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import time


def get_lapse_time(start_time, end_time):
    """
    格式化时间，可用于计算程序运行时间
    :param start_time: 开始时间
    :param end_time: 结束时间
    :return: 时分秒显示
    """
    start_num = 3600 * int(start_time[:2]) + 60 * int(start_time[2:4]) + int(start_time[-2:])
    end_num = 3600 * int(end_time[:2]) + 60 * int(end_time[2:4]) + int(end_time[-2:])
    hours = (end_num - start_num) // 3600
    minutes = ((end_num - start_num) % 3600) // 60
    seconds = ((end_num - start_num) % 3600) % 60
    gap_time = "%02d:%02d:%02d" % (hours, minutes, seconds)
    return gap_time


start = time.strftime("%H %M %S")
# MNIST数据存放的路径
file = "./MNIST_DATA"

# 导入数据
mnist = input_data.read_data_sets(file, one_hot=True)

# 模型的输入和输出
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# 模型的权重和偏移量
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# 创建Session
sess = tf.InteractiveSession()
# 初始化权重变量
sess.run(tf.global_variables_initializer())

y = tf.nn.softmax(tf.matmul(x, W) + b)

# 交叉熵
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

# 训练
train_start = time.strftime("%H %M %S")
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
for i in range(1000):
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})

train_end = time.strftime("%H %M %S")
print("训练用时：%s" % get_lapse_time(train_start, train_end))

# 测试
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

end = time.strftime("%H %M %S")
print("程序运行用时：%s" % get_lapse_time(start, end))
