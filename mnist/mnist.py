#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : Kerr
@License : Copyright(C) 2018, Get
@Contact : 905392619@qq.com
@Software: PyCharm
@File    : mnist.py
@Time    : 2018-09-013 16:46
@Desc    : 线性回归mnist手写体识别，准确率0.91
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

# 导入数据，input_data.read_data_sets()函数可以自动检测指定目录下是否存在MNIST数据，如果存在，就不会下载了。
mnist = input_data.read_data_sets(file, one_hot=True)

# 模型的输入和输出，x和y_表示输入和输出的占位符，可以在进行计算的时候进行赋值。
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# 模型的权重和偏移量，变量W和b是线性模型的参数，线性模型表达式为y_=x*W+b
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# 创建Session
sess = tf.InteractiveSession()
# 初始化权重变量
sess.run(tf.global_variables_initializer())

y = tf.nn.softmax(tf.matmul(x, W) + b)

# 为训练过程指定损失函数，损失函数是用来评估模型一次预测的好与坏的。在这里使用目标类别和预测类别之间的交叉熵作为我们的损失函数。
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

# 训练
train_start = time.strftime("%H %M %S")
# 使用TensorFlow内置的梯度下降来进行优化，即让损失函数的值下降，步长为0.01
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
for i in range(1000):
    batch = mnist.train.next_batch(50)   # 每次循环，都会从训练集中加载50个样本。
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})

train_end = time.strftime("%H %M %S")
print("训练用时：%s" % get_lapse_time(train_start, train_end))

# 测试
# 这里返回一个布尔数组，形如[True, False]
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# 将布尔数组转换为浮点数，并取平均值，如上布尔数组可以转换为[1, 0, 1]，计算平均值为0.667
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# 计算在测试数据上的准确率
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

end = time.strftime("%H %M %S")
print("程序运行用时：%s" % get_lapse_time(start, end))
