# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.contrib as tc

from tensorflow.python.profiler import model_analyzer
from tensorflow.python.profiler import option_builder
from tensorflow.python.client import timeline

import cv2

import numpy as np
from img_utils import *
from dnn_model import DnnModel

import matplotlib.pyplot as plt

numChannels = 3
imgWidth = 224
imgHeight = 224
numClasses = 1000

image_dir = '/home/ruiliu/Development/mtml-tf/dataset/imagenet10k'
#bin_dir = '/home/ruiliu/Development/mtml-tf/dataset/imagenet1k.bin'
label_path = '/home/ruiliu/Development/mtml-tf/dataset/imagenet10k-label.txt'
#bin_dir = '/tank/local/ruiliu/dataset/imagenet1k.bin'
#label_path = '/tank/local/ruiliu/dataset/imagenet1k-label.txt'
profile_dir = '/home/ruiliu/Development/mtml-tf/mt-perf/profile_dir/test'

#X_data = load_images_bin(bin_dir, numChannels, imgWidth, imgHeight)
#Y_data = load_labels_onehot(label_path, numClasses)

features = tf.placeholder(tf.float32, [None, imgHeight, imgWidth, numChannels])
labels = tf.placeholder(tf.int64, [None, numClasses])

image_name = 'ILSVRC2010_test_00030001.JPEG'

if __name__ == '__main__':

    ######################
    #test tf ops 
    ######################

    image_raw = tf.placeholder(tf.int64,shape=[500, 375, 3])
    trans_op = tf.image.resize_images(image_raw, (224, 224))
    #trans_op = tf.contrib.image.transform(image_raw,transforms=[1,0,0,0,1,0,0,0])
    img = cv2.imread(image_dir+'/'+image_name)
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    with tf.Session(config=config) as sess:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        sess.run(trans_op, feed_dict={image_raw:img}, options=run_options, run_metadata=run_metadata)
        trace = timeline.Timeline(step_stats=run_metadata.step_stats)
        trace_file = open('/home/ruiliu/Development/mtml-tf/mt-perf/profile_dir/test/tf_resize.json', 'w')
        trace_file.write(trace.generate_chrome_trace_format(show_memory=True))

    ######################
    #test tf.data api 
    ######################

    #with tf.Session(config=config) as sess:
    #    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    #    run_metadata = tf.RunMetadata()
    #    sess.run(tf.global_variables_initializer())
    #    my_profiler = model_analyzer.Profiler(graph=sess.graph)

    #    dataset_it = generate_image_dataset(image_dir,label_path)
    #    next_data = dataset_it.get_next()
    
    #    for i in range(10):
    #        sess.run(next_data)
    #        image, label = sess.run(next_data, options=run_options, run_metadata=run_metadata)
 
            #trace = timeline.Timeline(step_stats=run_metadata.step_stats)
            #trace_file = open('/home/ruiliu/Development/mtml-tf/mt-perf/profile_dir/test/sss.json', 'w')
            #trace_file.write(trace.generate_chrome_trace_format(show_memory=True))
            #my_profiler.add_step(step=i, run_meta=run_metadata)

    #profile_graph_builder = option_builder.ProfileOptionBuilder(option_builder.ProfileOptionBuilder.time_and_memory())
    #profile_graph_builder.with_timeline_output(timeline_file='/home/ruiliu/Development/mtml-tf/mt-perf/profile_dir/test/timeline.json')
    #profile_graph_builder.with_step([0,9])
    #my_profiler.profile_graph(profile_graph_builder.build())