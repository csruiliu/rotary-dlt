#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
sys.path.append('/home/ruiliu/Development/mtml-tf/models/research/slim')
sys.path.append('/home/ruiliu/Development/mtml-tf/models/official')

import tensorflow as tf
from rr_model import rr_net

dataset_size = 1000
loop_num = 1
batch_size = 10
batch_num = dataset_size / batch_size

tf.reset_default_graph()

net = tf.Graph()
with net.as_default():
    img_path = tf.placeholder(tf.string, ())
    batch = []

    for i in range(1, batch_size + 1):
        img_name = 'img' + str(i) + '.jpg'
        img = tf.image.decode_jpeg(tf.read_file(img_path + '/' + img_name))
        image = tf.expand_dims(img, 0)
        image = tf.cast(image, tf.float32) / 128. - 1
        #image = tf.cast(image, tf.float32) / 128. - 1
        image.set_shape((None, None, None, 3))
        image = tf.image.resize_images(image, (224, 224))


    inputs = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3], name='input-img')

