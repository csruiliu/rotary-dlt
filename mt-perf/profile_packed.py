# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.contrib as tc

from tensorflow.python.profiler import model_analyzer
from tensorflow.python.profiler import option_builder

import numpy as np

from img_utils import *
from dnn_model import DnnModel
from timeit import default_timer as timer
import datetime

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpuid', type=int, default=0, help='identify a GPU to run')
parser.add_argument('-bs', '--batchsize', type=int, default=10, help='identify the training batch size')
parser.add_argument('-iw', '--imgw', type=int, default=224, help='identify the weight of img')
parser.add_argument('-ih', '--imgh', type=int, default=224, help='identify the height of img')
parser.add_argument('-cls', '--clazz', type=int, default=1000, help='predication classes')
parser.add_argument('-ch', '--channel', type=int, default=3, help='identify the channel of input images')
parser.add_argument('-e', '--epoch', type=int, default=1, help='identify the epoch numbers')
parser.add_argument('-s', '--shuffle', action='store_true', default=False, help='use shuffle the batch input or not, default is sequential, if use different batch size, then this config will be ignored')
parser.add_argument('-d', '--diff', action='store_true', default=False, help='use different batch input for each model in the packed api, default is all the models in packed use all input batch')
args = parser.parse_args()

gpuId = args.gpuid
batchSize = args.batchsize
imgWidth = args.imgw
imgHeight = args.imgh
numClasses = args.clazz
numChannels = args.channel
numEpochs = args.epoch
isShuffle = args.shuffle
isDiffernetBatch = args.diff

modelCollection = []
modelEntityCollection = []
trainCollection = []
scheduleCollection = []
batchCollection = []

input_model_num = 5

#bin_dir = '/home/ruiliu/Development/mtml-tf/dataset/imagenet1k.bin'
#label_path = '/home/ruiliu/Development/mtml-tf/dataset/imagenet1k-label.txt'
bin_dir = '/tank/local/ruiliu/dataset/imagenet1k.bin'
label_path = '/tank/local/ruiliu/dataset/imagenet1k-label.txt'
X_data = load_images_bin(bin_dir, numChannels, imgWidth, imgHeight)
Y_data = load_labels_onehot(label_path, numClasses)

features = tf.placeholder(tf.float32, [None, imgHeight, imgWidth, numChannels])
labels = tf.placeholder(tf.int64, [None, numClasses])

profile_dir = '/home/ruiliu/Development/mtml-tf/mt-perf/profile_dir/test'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
with tf.contrib.tfprof.ProfileContext(profile_dir, trace_steps=range(0, 2), dump_steps=range(0, 2)) as pctx:
    with tf.Session(config=config) as sess:
        with tf.variable_scope('instance'):
            w = tf.get_variable('w', [3, 3, features.get_shape()[-1], 32],
              regularizer=tf.contrib.layers.l2_regularizer(1e-4),
              initializer=tf.truncated_normal_initializer(stddev=0.02))
            
            conv = tf.nn.conv2d(features, w, strides=[1, 2, 2, 1], padding='SAME')
            #conv = tc.layers.conv2d(features, 32, 3, 2, data_format="NHWC", normalizer_fn=tc.layers.batch_norm, normalizer_params={'is_training': True})
            #output = tf.layers.conv2d(inputs=features, filters=32, kernel_size=3, strides=2, data_format="channels_last",trainable=True)
        
        sess.run(tf.global_variables_initializer())
        batchSize = 50
        #for i in range(20):
        X_mini_batch_feed = X_data[0:batchSize,:,:,:]
        Y_mini_batch_feed = Y_data[0:batchSize,:]
        #X_mini_batch_feed = np.random.rand(1,224,224,3)
        #print(tf.shape(conv).eval(session=sess, feed_dict={features: X_mini_batch_feed}))
        #sess.run(tf.shape(output).eval(), feed_dict={features: X_mini_batch_feed})
        sess.run(conv, feed_dict={features: X_mini_batch_feed})
