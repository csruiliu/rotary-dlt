# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 2019
@author: Rui Liu
"""

import sys
sys.path.append('/home/rui/Development/mtml-tf/models/research/slim')
sys.path.append('/home/rui/Development/mtml-tf/models/official')
#sys.path.append('/home/ruiliu/Development/mtml-tf/models/research/slim')
#sys.path.append('/home/ruiliu/Development/mtml-tf/models/official')

import tensorflow as tf
from nets import resnet_v2
import preprocessing

class ResNet(object):

    def __init__(self, num_classes, is_training,fixed_resize_side=368,default_image_size=336):
        self._num_classes = num_classes
        self._is_training = is_training
        self._fixed_resize_side = fixed_resize_side
        self._default_image_size = default_image_size

    @property
    def num_classes(self):
        return self._num_classes

    def preprocess(self, inputs):
        preprocessed_inputs = preprocessing.preprocess_images(
            inputs, self._default_image_size, self._default_image_size,
            resize_side_min=self._fixed_resize_side,
            is_training=self._is_training, border_expand=True, normalize=False,
            preserving_aspect_ratio_resize=False)
        preprocessed_inputs = tf.cast(preprocessed_inputs, tf.float32)
        return preprocessed_inputs

    def predict(self, preprocessed_inputs):
        with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope()):
            net, endpoints = resnet_v2.resnet_v2_50(
                preprocessed_inputs, num_classes=None,
                is_training=self._is_training)
        net = tf.squeeze(net, axis=[1, 2])
        logits = tf.contrib.slim.fully_connected(net, num_outputs=self.num_classes,
                                      activation_fn=None, scope='Predict')
        prediction_dict = {'logits': logits}
        return prediction_dict

    def postprocess(self, prediction_dict):
        logits = prediction_dict['logits']
        logits = tf.nn.softmax(logits)
        classes = tf.argmax(logits, axis=1)
        postprocessed_dict = {'logits': logits,'classes': classes}
        return postprocessed_dict

    def loss(self, prediction_dict, groundtruth_lists):
        logits = prediction_dict['logits']
        tf.contrib.slim.losses.sparse_softmax_cross_entropy(
            logits=logits,
            labels=groundtruth_lists,
            scope='Loss')
        loss = tf.contrib.slim.losses.get_total_loss()
        loss_dict = {'loss': loss}
        return loss_dict

    def accuracy(self, postprocessed_dict, groundtruth_lists):
        classes = postprocessed_dict['classes']
        accuracy = tf.reduce_mean(tf.cast(tf.equal(classes, groundtruth_lists), dtype=tf.float32))
        return accuracy
