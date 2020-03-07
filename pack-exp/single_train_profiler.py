from __future__ import division
import tensorflow as tf
from tensorflow.python.client import timeline
import os
import numpy as np
from timeit import default_timer as timer

import config as cfg_yml
from dnn_model import DnnModel
from img_utils import *

def buildModel():
    model_name_abbr = np.random.choice(randSeed, 1, replace=False).tolist()
    dm = DnnModel(trainModel, str(model_name_abbr.pop()), trainNumLayer, imgHeight, imgWidth, numChannels, numClasses,
                  trainBatchSize, trainOptimizer, trainLearnRate, trainActivation, False)
    modelEntity = dm.getModelEntity()
    modelLogit = modelEntity.build(features)
    trainStep = modelEntity.train(modelLogit, labels)

    return trainStep

def profileStepRawImage():
    step_time = 0
    step_count = 0
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    image_list = sorted(os.listdir(image_path_raw))

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        num_batch = Y_data.shape[0] // trainBatchSize
        for e in range(numEpochs):
            for i in range(num_batch):
                print('epoch %d / %d, step %d / %d' % (e + 1, numEpochs, i + 1, num_batch))
                if (i + 1) % recordMarker == 0:
                    start_time = timer()
                    batch_offset = i * trainBatchSize
                    batch_end = (i + 1) * trainBatchSize
                    batch_list = image_list[batch_offset:batch_end]
                    X_mini_batch_feed = load_imagenet_raw(image_path_raw, batch_list, imgHeight, imgWidth)
                    Y_mini_batch_feed = Y_data[batch_offset:batch_end, :]

                    if useTimeline:
                        profile_path = cfg_yml.profile_path
                        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()
                        sess.run(trainOp, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed}, options=run_options, run_metadata=run_metadata)
                        trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                        trace_file = open(profile_path + '/' + str(trainModel) + '-' + str(trainBatchSize) + '-' + str(i) + '.json', 'w')
                        trace_file.write(trace.generate_chrome_trace_format(show_dataflow=True, show_memory=True))
                    else:
                        sess.run(trainOp, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})

                    end_time = timer()
                    dur_time = end_time - start_time
                    print("step time:", dur_time)
                    step_time += dur_time
                    step_count += 1
                else:
                    batch_offset = i * trainBatchSize
                    batch_end = (i + 1) * trainBatchSize
                    batch_list = image_list[batch_offset:batch_end]
                    X_mini_batch_feed = load_imagenet_raw(image_path_raw, batch_list, imgHeight, imgWidth)
                    Y_mini_batch_feed = Y_data[batch_offset:batch_end, :]
                    sess.run(trainOp, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})

        print("average step time:", step_time / step_count * 1000)


def profileEpochRawImage():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    image_list = sorted(os.listdir(image_path_raw))

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        num_batch = Y_data.shape[0] // trainBatchSize
        start_time = timer()
        for e in range(numEpochs):
            for i in range(num_batch):
                print('epoch %d / %d, step %d / %d' % (e + 1, numEpochs, i + 1, num_batch))
                batch_offset = i * trainBatchSize
                batch_end = (i + 1) * trainBatchSize
                batch_list = image_list[batch_offset:batch_end]
                X_mini_batch_feed = load_imagenet_raw(image_path_raw, batch_list, imgHeight, imgWidth)
                Y_mini_batch_feed = Y_data[batch_offset:batch_end, :]
                sess.run(trainOp, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})
        end_time = timer()
        dur_time = end_time - start_time
        print("training time ({} epoch): {} seconds".format(numEpochs, dur_time))


def profileStep():
    step_time = 0
    step_count = 0
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        num_batch = Y_data.shape[0] // trainBatchSize
        for e in range(numEpochs):
            for i in range(num_batch):
                print('epoch %d / %d, step %d / %d' % (e + 1, numEpochs, i + 1, num_batch))
                if (i + 1) % recordMarker == 0:
                    start_time = timer()
                    batch_offset = i * trainBatchSize
                    batch_end = (i + 1) * trainBatchSize
                    X_mini_batch_feed = X_data[batch_offset:batch_end, :, :, :]
                    Y_mini_batch_feed = Y_data[batch_offset:batch_end, :]

                    if useTimeline:
                        profile_path = cfg_yml.profile_path
                        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()
                        sess.run(trainOp, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed}, options=run_options, run_metadata=run_metadata)
                        trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                        trace_file = open(profile_path + '/' + str(trainModel) + '-' + str(trainBatchSize) + '-' + str(i) + '.json', 'w')
                        trace_file.write(trace.generate_chrome_trace_format(show_dataflow=True, show_memory=True))
                    else:
                        sess.run(trainOp, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})

                    end_time = timer()
                    dur_time = end_time - start_time
                    print("step time:", dur_time)
                    step_time += dur_time
                    step_count += 1
                else:
                    batch_offset = i * trainBatchSize
                    batch_end = (i + 1) * trainBatchSize
                    X_mini_batch_feed = X_data[batch_offset:batch_end, :, :, :]
                    Y_mini_batch_feed = Y_data[batch_offset:batch_end, :]
                    sess.run(trainOp, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})

        print(step_time)
        print(step_count)
        print("average step time:", step_time / step_count * 1000)


def profileEpoch():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        num_batch = Y_data.shape[0] // trainBatchSize
        start_time = timer()
        for e in range(numEpochs):
            for i in range(num_batch):
                print('epoch %d / %d, step %d / %d' % (e + 1, numEpochs, i + 1, num_batch))
                batch_offset = i * trainBatchSize
                batch_end = (i + 1) * trainBatchSize
                X_mini_batch_feed = X_data[batch_offset:batch_end, :, :, :]
                Y_mini_batch_feed = Y_data[batch_offset:batch_end, :]
                sess.run(trainOp, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})

        end_time = timer()
        dur_time = end_time - start_time
        print("training time ({} epoch): {} seconds".format(numEpochs, dur_time))



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
    # Build
    #########################

    features = tf.placeholder(tf.float32, [None, imgWidth, imgHeight, numChannels])
    labels = tf.placeholder(tf.int64, [None, numClasses])

    trainOp = buildModel()

    #########################
    # Train
    #########################

    image_path_raw = cfg_yml.imagenet_t1k_img_path
    image_path_bin = cfg_yml.imagenet_t1k_bin_path
    label_path = cfg_yml.imagenet_t1k_label_path

    if useRawImage:
        Y_data = load_imagenet_labels_onehot(label_path, numClasses)

        if measureStep:
            profileStepRawImage()
        else:
            profileEpochRawImage()
    else:
        X_data = load_imagenet_bin(image_path_bin, numChannels, imgWidth, imgHeight)
        Y_data = load_imagenet_labels_onehot(label_path, numClasses)

        if measureStep:
            profileStep()
        else:
            profileEpoch()
