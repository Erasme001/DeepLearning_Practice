#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : Kerr
@Contact : 905392619@qq.com
@Software: PyCharm
@File    : delete_broken_img.py
@Time    : 2018-10-10 10:52
@Desc    : 去除图像中的异常或单通道图像
"""
import tensorflow as tf
from glob import glob
import os
import argparse
import logging
from PIL import Image


def glob_all(dir_path):
    """
    递归取出dir_path下的所有jpg文件
    :param dir_path: 文件路径
    :return: 图片列表
    """
    pic_list = glob(os.path.join(dir_path, '*.jpg'))
    inside = os.listdir(dir_path)
    for dir_name in inside:
        if os.path.isdir(os.path.join(dir_path, dir_name)):
            pic_list.extend(glob_all(os.path.join(dir_path, dir_name)))
    return pic_list


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--dir-path', default='data/')
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    all_pic_list = glob_all(args.dir_path)
    for i, img_path in enumerate(all_pic_list):
        try:
            sess = tf.Session()
            with open(img_path, 'rb') as f:
                img_byte = f.read()
                # 判断图像是否损坏，如果损坏就会抛出异常
                img = tf.image.decode_jpeg(img_byte)
                data = sess.run(img)
                # 判断图像是否拥有三个通道，如果通道数不为3，就抛出异常
                if data.shape[2] != 3:
                    print(data.shape)
                    raise Exception
            tf.reset_default_graph()
            img = Image.open(img_path)
        except Exception:
            # 检测到异常就删掉图片
            logging.warning('%s has broken. Delete it.' % img_path)
            os.remove(img_path)
        if (i + 1) % 1000 == 0:
            logging.info('Processing %d / %d.' % (i + 1, len(all_pic_list)))
