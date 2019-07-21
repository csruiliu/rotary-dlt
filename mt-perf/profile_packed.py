# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.contrib as tc

import numpy as np
from img_utils import *
from dnn_model import DnnModel

from tensorflow.python.profiler import model_analyzer
from tensorflow.python.profiler import option_builder


numChannels = 3
imgWidth = 224
imgHeight = 224
numClasses = 1000


bin_dir = '/home/ruiliu/Development/mtml-tf/dataset/imagenet1k.bin'
label_path = '/home/ruiliu/Development/mtml-tf/dataset/imagenet1k-label.txt'
#bin_dir = '/tank/local/ruiliu/dataset/imagenet1k.bin'
#label_path = '/tank/local/ruiliu/dataset/imagenet1k-label.txt'
X_data = load_images_bin(bin_dir, numChannels, imgWidth, imgHeight)
Y_data = load_labels_onehot(label_path, numClasses)

features = tf.placeholder(tf.float32, [None, imgHeight, imgWidth, numChannels])
labels = tf.placeholder(tf.int64, [None, numClasses])

profile_dir = '/home/ruiliu/Development/mtml-tf/mt-perf/profile_dir/test'
    
if __name__ == '__main__':
    
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        model_profiler = model_analyzer.Profiler(graph=sess.graph)
        for e in range(numEpochs):
            for i in range(batchSize):
                print('epoch %d / %d, step %d / %d' %(e+1, numEpochs, i+1, batchSize))    
                X_mini_batch_feed = X_data[i:i+batchSize,:,:,:]
                Y_mini_batch_feed = Y_data[i:i+batchSize,:]
                sess.run(logit, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed}, options=run_options, run_metadata=run_metadata)
                model_profiler.add_step(step=i, run_meta=run_metadata) 

        profile_graph_opts_builder = option_builder.ProfileOptionBuilder(option_builder.ProfileOptionBuilder.time_and_memory())
        #profile_graph_opts_builder.with_timeline_output(timeline_file='/tank/ruiliu/mtml-tf/mt-perf/profile_dir/mobile-m1-b10/model_profiler.json')
        profile_graph_opts_builder.with_timeline_output(timeline_file='/home/ruiliu/Development/mtml-tf/mt-perf/profile_dir/test/model_profiler.json')
        model_profiler.profile_graph(profile_graph_opts_builder.build())