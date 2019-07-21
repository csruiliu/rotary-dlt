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
bin_dir = '/home/ruiliu/Development/mtml-tf/dataset/imagenet1k.bin'
label_path = '/home/ruiliu/Development/mtml-tf/dataset/imagenet1k-label.txt'
#bin_dir = '/tank/local/ruiliu/dataset/imagenet1k.bin'
#label_path = '/tank/local/ruiliu/dataset/imagenet1k-label.txt'
X_data = load_images_bin(bin_dir, numChannels, imgWidth, imgHeight)
Y_data = load_labels_onehot(label_path, numClasses)

features = tf.placeholder(tf.float32, [None, imgHeight, imgWidth, numChannels])
labels = tf.placeholder(tf.int64, [None, numClasses])

profile_dir = '/home/ruiliu/Development/mtml-tf/mt-perf/profile_dir/test'

image_name = 'ILSVRC2010_test_00030001.JPEG'

def read_image():
    pass

img = cv2.imread(image_dir+'/'+image_name)
#print(type(img))
if __name__ == '__main__':

    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    image_raw = tf.placeholder(tf.int64,shape=[500, 375, 3])
    trans_op = tf.image.resize_images(image_raw, (224, 224))
    #trans_op = tf.contrib.image.transform(image_raw,transforms=[1,0,0,0,1,0,0,0])

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    with tf.device('/device:GPU:0'):
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            model_profiler = model_analyzer.Profiler(graph=sess.graph)
            
            trans_img = sess.run(trans_op, feed_dict={image_raw: img}, options=run_options, run_metadata=run_metadata)
            print(type(trans_img))
            #imgplot = plt.imshow(trans_img)
            #plt.show()
            #cv2.imshow('img',trans_img)
            
            trace = timeline.Timeline(step_stats=run_metadata.step_stats)
            trace_file = open('/home/ruiliu/Development/mtml-tf/mt-perf/profile_dir/test/sss.json', 'w')
            trace_file.write(trace.generate_chrome_trace_format(show_memory=True))
            
            #model_profiler.add_step(step=0, run_meta=run_metadata) 
            #profile_graph_opts_builder = option_builder.ProfileOptionBuilder(option_builder.ProfileOptionBuilder.time_and_memory())
            #profile_graph_opts_builder.with_timeline_output(timeline_file='/tank/ruiliu/mtml-tf/mt-perf/profile_dir/mobile-m1-b10/model_profiler.json')
            #profile_graph_opts_builder.with_timeline_output(timeline_file='/home/ruiliu/Development/mtml-tf/mt-perf/profile_dir/test/sss.json')
            #model_profiler.profile_graph(profile_graph_opts_builder.build())
        