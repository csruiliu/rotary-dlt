from __future__ import division
import tensorflow as tf
from tensorflow.python.client import timeline
import os
import numpy as np
from timeit import default_timer as timer

import config as cfg_yml
from dnn_model import DnnModel
from img_utils import *


def buildModel(train_model, num_layer, input_w, input_h, num_channels, num_classes, batch_size,
               optimizer, learning_rate, activation, batch_padding, rand_seed, features_ph, labels_ph):
    model_name_abbr = np.random.choice(rand_seed, 1, replace=False).tolist()
    dm = DnnModel(train_model, str(model_name_abbr.pop()), num_layer, input_w, input_h, num_channels, num_classes,
                  batch_size, optimizer, learning_rate, activation, batch_padding)
    modelEntity = dm.getModelEntity()
    modelLogit = modelEntity.build(features_ph)
    trainStep = modelEntity.train(modelLogit, labels_ph)

    return trainStep

def profileStep(train_step, num_epoch, X_train, Y_train, record_marker, use_timeline):
    step_time = 0
    step_count = 0
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        num_batch = Y_train.shape[0] // trainBatchSize
        for e in range(num_epoch):
            for i in range(num_batch):
                print('epoch %d / %d, step %d / %d' % (e + 1, num_epoch, i + 1, num_batch))
                if (i + 1) % record_marker == 0:
                    start_time = timer()
                    batch_offset = i * trainBatchSize
                    batch_end = (i + 1) * trainBatchSize
                    X_mini_batch_feed = X_train[batch_offset:batch_end, :, :, :]
                    Y_mini_batch_feed = Y_train[batch_offset:batch_end, :]

                    if use_timeline:
                        profile_path = cfg_yml.profile_path
                        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()
                        sess.run(train_step, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed}, options=run_options, run_metadata=run_metadata)
                        trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                        trace_file = open(profile_path + '/' + str(trainModel) + '-' + str(trainBatchSize) + '-' + str(i) + '.json', 'w')
                        trace_file.write(trace.generate_chrome_trace_format(show_dataflow=True, show_memory=True))
                    else:
                        sess.run(train_step, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})

                    end_time = timer()
                    dur_time = end_time - start_time
                    print("step time:", dur_time)
                    step_time += dur_time
                    step_count += 1
                else:
                    batch_offset = i * trainBatchSize
                    batch_end = (i + 1) * trainBatchSize
                    X_mini_batch_feed = X_train[batch_offset:batch_end, :, :, :]
                    Y_mini_batch_feed = Y_train[batch_offset:batch_end, :]
                    sess.run(train_step, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})

        print(step_time)
        print(step_count)
        print("average step time:", step_time / step_count * 1000)


def profileEpoch(train_step, num_epoch, X_train, Y_train):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        num_batch = Y_train.shape[0] // trainBatchSize
        start_time = timer()
        for e in range(num_epoch):
            for i in range(num_batch):
                print('epoch %d / %d, step %d / %d' % (e + 1, num_epoch, i + 1, num_batch))
                batch_offset = i * trainBatchSize
                batch_end = (i + 1) * trainBatchSize
                X_mini_batch_feed = X_train[batch_offset:batch_end, :, :, :]
                Y_mini_batch_feed = Y_train[batch_offset:batch_end, :]
                sess.run(train_step, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})

        end_time = timer()
        dur_time = end_time - start_time
        print("training time ({} epoch): {} seconds".format(num_epoch, dur_time))


def profileStepRawImage(trainStep, num_epoch, X_train_path, Y_train, record_marker, use_timeline):
    step_time = 0
    step_count = 0
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    image_list = sorted(os.listdir(X_train_path))

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        num_batch = Y_train.shape[0] // trainBatchSize
        for e in range(num_epoch):
            for i in range(num_batch):
                print('epoch %d / %d, step %d / %d' % (e + 1, num_epoch, i + 1, num_batch))
                if (i + 1) % record_marker == 0:
                    start_time = timer()
                    batch_offset = i * trainBatchSize
                    batch_end = (i + 1) * trainBatchSize
                    batch_list = image_list[batch_offset:batch_end]
                    X_mini_batch_feed = load_imagenet_raw(X_train_path, batch_list, imgHeight, imgWidth)
                    Y_mini_batch_feed = Y_train[batch_offset:batch_end, :]

                    if use_timeline:
                        profile_path = cfg_yml.profile_path
                        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()
                        sess.run(trainStep, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed}, options=run_options, run_metadata=run_metadata)
                        trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                        trace_file = open(profile_path + '/' + str(trainModel) + '-' + str(trainBatchSize) + '-' + str(i) + '.json', 'w')
                        trace_file.write(trace.generate_chrome_trace_format(show_dataflow=True, show_memory=True))
                    else:
                        sess.run(trainStep, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})

                    end_time = timer()
                    dur_time = end_time - start_time
                    print("step time:", dur_time)
                    step_time += dur_time
                    step_count += 1
                else:
                    batch_offset = i * trainBatchSize
                    batch_end = (i + 1) * trainBatchSize
                    batch_list = image_list[batch_offset:batch_end]
                    X_mini_batch_feed = load_imagenet_raw(X_train_path, batch_list, imgHeight, imgWidth)
                    Y_mini_batch_feed = Y_train[batch_offset:batch_end, :]
                    sess.run(trainStep, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})

        print(step_time)
        print(step_count)
        print("average step time:", step_time / step_count * 1000)

def profileEpochRawImage(trainStep, num_epoch, X_train_path, Y_train):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    image_list = sorted(os.listdir(X_train_path))
    
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        num_batch = Y_train.shape[0] // trainBatchSize
        start_time = timer()
        for e in range(num_epoch):
            for i in range(num_batch):
                print('epoch %d / %d, step %d / %d' %(e+1, num_epoch, i+1, num_batch))
                batch_offset = i * trainBatchSize
                batch_end = (i+1) * trainBatchSize
                batch_list = image_list[batch_offset:batch_end]  
                X_mini_batch_feed = load_imagenet_raw(X_train_path, batch_list, imgHeight, imgWidth)
                Y_mini_batch_feed = Y_train[batch_offset:batch_end,:]
                sess.run(trainStep, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})
        end_time = timer()
        dur_time = end_time - start_time
        print("training time ({} epoch): {} seconds".format(num_epoch, dur_time))


if __name__ == '__main__':

    #########################
    # Parameters
    #########################

    imgWidth = cfg_yml.single_img_width
    imgHeight = cfg_yml.single_img_height
    numChannels = cfg_yml.single_num_channels
    numClasses = cfg_yml.single_num_classes
    numEpochs = cfg_yml.single_num_epoch
    trainModel = cfg_yml.single_model_type
    trainBatchSize = cfg_yml.single_batch_size
    trainOptimizer = cfg_yml.single_opt
    randSeed = cfg_yml.single_rand_seed
    trainNumLayer = cfg_yml.single_num_layer
    trainLearnRate = cfg_yml.single_learning_rate
    trainActivation = cfg_yml.single_activation

    useRawImage = cfg_yml.single_use_raw_image
    measureStep = cfg_yml.single_measure_step
    useCPU = cfg_yml.single_use_cpu
    recordMarker = cfg_yml.single_record_marker
    useTimeline = cfg_yml.single_use_tb_timeline

    if useCPU:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    #########################
    # Build and Train
    #########################

    features = tf.placeholder(tf.float32, [None, imgWidth, imgHeight, numChannels])
    labels = tf.placeholder(tf.int64, [None, numClasses])

    trainOp = buildModel(trainModel, trainNumLayer, imgHeight, imgWidth, numChannels, numClasses, trainBatchSize,
                           trainOptimizer, trainLearnRate, trainActivation, False, randSeed, features, labels)

    if useRawImage:
        image_path = cfg_yml.imagenet_t1k_img_path
        label_path = cfg_yml.imagenet_t1k_label_path
        Y_data = load_imagenet_labels_onehot(label_path, numClasses)

        if measureStep:
            profileStepRawImage(trainOp, numEpochs, image_path, Y_data, recordMarker, useTimeline)
        else:
            profileEpochRawImage(trainOp, numEpochs, image_path, Y_data)
    else:
        image_path = cfg_yml.imagenet_t1k_bin_path
        label_path = cfg_yml.imagenet_t1k_label_path
        X_data = load_imagenet_bin(image_path, numChannels, imgWidth, imgHeight)
        Y_data = load_imagenet_labels_onehot(label_path, numClasses)

        if measureStep:
            profileStep(trainOp, numEpochs, X_data, Y_data, recordMarker, useTimeline)
        else:
            profileEpoch(trainOp, numEpochs, X_data, Y_data)
