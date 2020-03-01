from __future__ import division
import tensorflow as tf
from tensorflow.python.client import timeline
import os
import numpy as np
from timeit import default_timer as timer

import config as cfg_yml
from dnn_model import DnnModel
from img_utils import *


def buildModels(trainModel, num_layer, input_w, input_h, num_channels, num_classes, batch_size, optimizer,
                learning_rate, activation, batch_padding, rand_seed):

    model_name_abbr = np.random.choice(rand_seed, len(trainModel), replace=False).tolist()

    if sameInput:
        trainOpCollection = []
        features = tf.placeholder(tf.float32, [None, imgWidth, imgHeight, numChannels])
        labels = tf.placeholder(tf.int64, [None, numClasses])

        for idx, mt in enumerate(trainModel):
            dm = DnnModel(mt, str(model_name_abbr.pop()), num_layer[idx], input_h, input_w, num_channels, num_classes,
                          batch_size[idx], optimizer[idx], learning_rate[idx], activation[idx], batch_padding)
            modelEntity = dm.getModelEntity()
            modelLogit = modelEntity.build(features)
            trainStep = modelEntity.train(modelLogit, labels)
            trainOpCollection.append(trainStep)

        return trainOpCollection, features, labels

    else:
        trainOpCollection = []
        featuresCollection = []
        labelsCollection = []
        names = locals()
        for idx, mt in range(trainModel):
            names['features' + str(idx)] = tf.placeholder(tf.float32, [None, imgWidth, imgHeight, numChannels])
            names['labels' + str(idx)] = tf.placeholder(tf.int64, [None, numClasses])

            featuresCollection.append(names['features' + str(idx)])
            labelsCollection.append(names['labels' + str(idx)])

            dm = DnnModel(mt, str(model_name_abbr.pop()), num_layer[idx], input_h, input_w, num_channels, num_classes,
                          batch_size[idx], optimizer[idx], learning_rate[idx], activation[idx], batch_padding)

            modelEntity = dm.getModelEntity()
            modelLogit = modelEntity.build(names['features' + str(idx)])
            trainStep = modelEntity.train(modelLogit, names['labels'+str(idx)])
            trainOpCollection.append(trainStep)

        return trainOpCollection, featuresCollection, labelsCollection


def profileStepRawImageSameInput(trainOpPack, num_epoch, X_train_path, Y_train, maxBatchSize, record_marker, use_timeline):
    step_time = 0
    step_count = 0
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    image_list = sorted(os.listdir(X_train_path))

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        num_batch = Y_train.shape[0] // maxBatchSize

        for e in range(num_epoch):
            for i in range(num_batch):
                print('epoch %d / %d, step %d / %d' % (e + 1, num_epoch, i + 1, num_batch))
                if (i + 1) % record_marker == 0:
                    start_time = timer()
                    batch_offset = i * maxBatchSize
                    batch_end = (i + 1) * maxBatchSize
                    batch_list = image_list[batch_offset:batch_end]
                    X_mini_batch_feed = load_imagenet_raw(X_train_path, batch_list, imgHeight, imgWidth)
                    Y_mini_batch_feed = Y_train[batch_offset:batch_end, :]

                    if use_timeline:
                        profile_path = cfg_yml.profile_path
                        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()
                        sess.run(trainOpPack, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed}, options=run_options, run_metadata=run_metadata)
                        trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                        trace_file = open(profile_path + '/' + str(trainModel) + '-' + str(trainBatchSize) + '-' + str(i) + '.json', 'w')
                        trace_file.write(trace.generate_chrome_trace_format(show_dataflow=True, show_memory=True))
                    else:
                        sess.run(trainOpPack, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})

                    end_time = timer()
                    dur_time = end_time - start_time
                    print("step time:", dur_time)
                    step_time += dur_time
                    step_count += 1
                else:
                    batch_offset = i * maxBatchSize
                    batch_end = (i + 1) * maxBatchSize
                    batch_list = image_list[batch_offset:batch_end]
                    X_mini_batch_feed = load_imagenet_raw(X_train_path, batch_list, imgHeight, imgWidth)
                    Y_mini_batch_feed = Y_train[batch_offset:batch_end, :]
                    sess.run(trainOpPack, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})


def profileStepRawImageDiffInput(trainOpPack, num_epoch, X_train_path, Y_train, maxBatchSize, record_marker, use_timeline):
    step_time = 0
    step_count = 0
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    image_list = sorted(os.listdir(X_train_path))
    input_model_num = len(trainOpPack)
    names = locals()
    input_dict = []

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        num_batch = Y_train.shape[0] // maxBatchSize
        num_batch_list = np.arange(num_batch)

        for e in range(num_epoch):
            for i in range(num_batch):

                if (i + 1) % record_marker == 0:
                    start_time = timer()
                    for ridx in range(input_model_num):
                        rand_idx = int(np.random.choice(num_batch_list, 1))
                        batch_offset = rand_idx * maxBatchSize
                        batch_end = (rand_idx + 1) * maxBatchSize
                        batch_list = image_list[batch_offset:batch_end]
                        names['X_mini_batch_feed' + str(ridx)] = load_imagenet_raw(X_train_path, batch_list, imgHeight, imgWidth)
                        names['Y_mini_batch_feed' + str(ridx)] = Y_train[batch_offset:batch_end, :]
                        input_dict[names['features' + str(ridx)]] = names['X_mini_batch_feed' + str(ridx)]
                        input_dict[names['labels' + str(ridx)]] = names['Y_mini_batch_feed' + str(ridx)]
                    sess.run(trainOpPack, feed_dict=input_dict)
                    end_time = timer()
                    dur_time = end_time - start_time
                    print("step time:", dur_time)
                    step_time += dur_time
                    step_count += 1
                else:
                    for ridx in range(input_model_num):
                        rand_idx = int(np.random.choice(num_batch_list, 1))
                        batch_offset = rand_idx * maxBatchSize
                        batch_end = (rand_idx + 1) * maxBatchSize
                        batch_list = image_list[batch_offset:batch_end]
                        names['X_mini_batch_feed' + str(ridx)] = load_imagenet_raw(X_train_path, batch_list, imgHeight, imgWidth)
                        names['Y_mini_batch_feed' + str(ridx)] = Y_train[batch_offset:batch_end, :]
                        input_dict[names['features' + str(ridx)]] = names['X_mini_batch_feed' + str(ridx)]
                        input_dict[names['labels' + str(ridx)]] = names['Y_mini_batch_feed' + str(ridx)]
                    sess.run(trainOpPack, feed_dict=input_dict)


def profileEpochRawImageSameInput(trainOpPack, num_epoch, X_train_path, Y_train, maxBatchSize):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    image_list = sorted(os.listdir(X_train_path))

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        num_batch = Y_train.shape[0] // maxBatchSize

        start_time = timer()
        for e in range(num_epoch):
            for i in range(num_batch):
                print('epoch %d / %d, step %d / %d' % (e + 1, num_epoch, i + 1, num_batch))

                start_time = timer()
                batch_offset = i * maxBatchSize
                batch_end = (i + 1) * maxBatchSize
                batch_list = image_list[batch_offset:batch_end]
                X_mini_batch_feed = load_imagenet_raw(X_train_path, batch_list, imgHeight, imgWidth)
                Y_mini_batch_feed = Y_train[batch_offset:batch_end, :]
                sess.run(trainOpPack, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})

        end_time = timer()
        dur_time = end_time - start_time
        print("training time ({} epoch): {} seconds".format(num_epoch, dur_time))


def profileEpochRawImageDiffInput(trainOpPack, num_epoch, X_train_path, Y_train, maxBatchSize):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    image_list = sorted(os.listdir(X_train_path))
    input_model_num = len(trainOpPack)
    names = locals()
    input_dict = []

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        num_batch = Y_train.shape[0] // maxBatchSize
        num_batch_list = np.arange(num_batch)

        start_time = timer()
        for e in range(num_epoch):
            for i in range(num_batch):
                for ridx in range(input_model_num):
                    rand_idx = int(np.random.choice(num_batch_list, 1))
                    batch_offset = rand_idx * maxBatchSize
                    batch_end = (rand_idx + 1) * maxBatchSize
                    batch_list = image_list[batch_offset:batch_end]
                    names['X_mini_batch_feed' + str(ridx)] = load_imagenet_raw(X_train_path, batch_list, imgHeight, imgWidth)
                    names['Y_mini_batch_feed' + str(ridx)] = Y_train[batch_offset:batch_end, :]
                    input_dict[names['features' + str(ridx)]] = names['X_mini_batch_feed' + str(ridx)]
                    input_dict[names['labels' + str(ridx)]] = names['Y_mini_batch_feed' + str(ridx)]
                sess.run(trainOpPack, feed_dict=input_dict)

        end_time = timer()
        dur_time = end_time - start_time
        print("training time ({} epoch): {} seconds".format(num_epoch, dur_time))


def profileStepSameInput(trainOpPack, num_epoch, X_train, Y_train, record_marker, use_timeline):
    step_time = 0
    step_count = 0
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        num_batch = Y_train.shape[0] // maxBatchSize
        num_batch_list = np.arange(num_batch)

        for e in range(num_epoch):
            for i in range(num_batch):
                if (i + 1) % record_marker == 0:
                    start_time = timer()
                    batch_offset = i * maxBatchSize
                    batch_end = (i + 1) * maxBatchSize
                    X_mini_batch_feed = X_train[batch_offset:batch_end, :, :, :]
                    Y_mini_batch_feed = Y_train[batch_offset:batch_end, :]

                    if use_timeline:
                        profile_path = cfg_yml.profile_path
                        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()
                        sess.run(trainOpPack, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed}, options=run_options, run_metadata=run_metadata)
                        trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                        trace_file = open(profile_path + '/' + str(trainModel) + '-' + str(trainBatchSize) + '-' + str(i) + '.json', 'w')
                        trace_file.write(trace.generate_chrome_trace_format(show_dataflow=True, show_memory=True))
                    else:
                        sess.run(trainOpPack, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})

                    end_time = timer()
                    dur_time = end_time - start_time
                    print("step time:", dur_time)
                    step_time += dur_time
                    step_count += 1

                else:
                    batch_offset = i * maxBatchSize
                    batch_end = (i + 1) * maxBatchSize
                    X_mini_batch_feed = X_train[batch_offset:batch_end, :, :, :]
                    Y_mini_batch_feed = Y_train[batch_offset:batch_end, :]
                    sess.run(trainOpPack, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})


def profileStepDiffInput(trainOpPack, num_epoch, X_train, Y_train, record_marker, use_timeline):
    step_time = 0
    step_count = 0
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    input_model_num = len(trainOpPack)
    names = locals()
    input_dict = []

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        num_batch = Y_train.shape[0] // maxBatchSize
        num_batch_list = np.arange(num_batch)

        for e in range(num_epoch):
            for i in range(num_batch):
                if (i + 1) % record_marker == 0:
                    start_time = timer()
                    for ridx in range(input_model_num):
                        rand_idx = int(np.random.choice(num_batch_list, 1))
                        batch_offset = rand_idx * maxBatchSize
                        batch_end = (rand_idx + 1) * maxBatchSize
                        names['X_mini_batch_feed' + str(ridx)] = X_train[batch_offset:batch_end, :, :, :]
                        names['Y_mini_batch_feed' + str(ridx)] = Y_train[batch_offset:batch_end, :]
                        input_dict[names['features' + str(ridx)]] = names['X_mini_batch_feed' + str(ridx)]
                        input_dict[names['labels' + str(ridx)]] = names['Y_mini_batch_feed' + str(ridx)]

                    sess.run(trainOpPack, feed_dict=input_dict)
                    end_time = timer()
                    dur_time = end_time - start_time
                    print("step time:", dur_time)
                    step_time += dur_time
                    step_count += 1

                else:
                    for ridx in range(input_model_num):
                        rand_idx = int(np.random.choice(num_batch_list, 1))
                        batch_offset = rand_idx * maxBatchSize
                        batch_end = (rand_idx + 1) * maxBatchSize
                        names['X_mini_batch_feed' + str(ridx)] = X_train[batch_offset:batch_end, :, :, :]
                        names['Y_mini_batch_feed' + str(ridx)] = Y_train[batch_offset:batch_end, :]
                        input_dict[names['features' + str(ridx)]] = names['X_mini_batch_feed' + str(ridx)]
                        input_dict[names['labels' + str(ridx)]] = names['Y_mini_batch_feed' + str(ridx)]
                    sess.run(trainOpPack, feed_dict=input_dict)

def profileEpochSameInput(trainOp, num_epoch, X_train, Y_train):
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
                sess.run(trainOp, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})

        end_time = timer()
        dur_time = end_time - start_time
        print("training time ({} epoch): {} seconds".format(num_epoch, dur_time))


def profileEpochDiffInput(trainOp, num_epoch, X_train, Y_train):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True


    input_model_num = len(trainOpPack)
    names = locals()
    input_dict = []

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        num_batch = Y_train.shape[0] // trainBatchSize
        num_batch_list = np.arange(num_batch)

        start_time = timer()
        for e in range(num_epoch):
            for i in range(num_batch):
                print('epoch %d / %d, step %d / %d' % (e + 1, num_epoch, i + 1, num_batch))
                for ridx in range(input_model_num):
                    rand_idx = int(np.random.choice(num_batch_list, 1))
                    batch_offset = rand_idx * maxBatchSize
                    batch_end = (rand_idx + 1) * maxBatchSize
                    names['X_mini_batch_feed' + str(ridx)] = X_train[batch_offset:batch_end, :, :, :]
                    names['Y_mini_batch_feed' + str(ridx)] = Y_train[batch_offset:batch_end, :]
                    input_dict[names['features' + str(ridx)]] = names['X_mini_batch_feed' + str(ridx)]
                    input_dict[names['labels' + str(ridx)]] = names['Y_mini_batch_feed' + str(ridx)]
                sess.run(trainOpPack, feed_dict=input_dict)

if __name__ == '__main__':

    #########################
    # Parameters
    #########################

    imgWidth = cfg_yml.pack_img_width
    imgHeight = cfg_yml.pack_img_height
    numChannels = cfg_yml.pack_num_channels
    numClasses = cfg_yml.pack_num_classes
    numEpochs = cfg_yml.pack_num_epoch
    trainModel = cfg_yml.pack_model_type
    trainBatchSize = cfg_yml.pack_batch_size
    trainOptimizer = cfg_yml.pack_opt
    randSeed = cfg_yml.pack_rand_seed
    trainNumLayer = cfg_yml.pack_num_layer
    trainLearnRate = cfg_yml.pack_learning_rate
    trainActivation = cfg_yml.pack_activation

    batchPadding = cfg_yml.pack_batch_padding
    useRawImage = cfg_yml.pack_use_raw_image
    measureStep = cfg_yml.pack_measure_step
    useCPU = cfg_yml.pack_use_cpu
    sameInput = cfg_yml.pack_same_input
    recordMarker = cfg_yml.pack_record_marker
    useTimeline = cfg_yml.pack_use_tb_timeline


    maxBatchSize = max(trainBatchSize)

    if useCPU:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    #########################
    # Build and Train
    #########################

    trainOpPack, features, labels = buildModels(trainModel, trainNumLayer, imgHeight, imgWidth, numChannels, numClasses, trainBatchSize,
                              trainOptimizer, trainLearnRate, trainActivation, batchPadding, randSeed)

    image_path_raw = cfg_yml.imagenet_t1k_img_path
    image_path_bin = cfg_yml.imagenet_t1k_bin_path
    label_path = cfg_yml.imagenet_t1k_label_path


    if useRawImage:
        Y_data = load_imagenet_labels_onehot(label_path, numClasses)

        if measureStep:
            if sameInput:
                profileStepRawImageSameInput(trainOpPack, numEpochs, image_path_raw, Y_data, maxBatchSize, recordMarker,
                                             useTimeline)
            else:
                profileStepRawImageDiffInput(trainOpPack, numEpochs, image_path_raw, Y_data, maxBatchSize, recordMarker,
                                             useTimeline)
        else:
            if sameInput:
                profileEpochRawImageSameInput(trainOpPack, numEpochs, image_path_raw, Y_data, maxBatchSize)
            else:
                profileEpochRawImageDiffInput(trainOpPack, numEpochs, image_path_raw, Y_data, maxBatchSize)
    else:
        X_data = load_imagenet_bin(image_path_bin, numChannels, imgWidth, imgHeight)
        Y_data = load_imagenet_labels_onehot(label_path, numClasses)

        if measureStep:
            if sameInput:
                profileStepSameInput(trainOpPack, numEpochs, X_data, Y_data, recordMarker, useTimeline)
            else:
                profileStepDiffInput(trainOpPack, numEpochs, X_data, Y_data, recordMarker, useTimeline)
        else:
            if sameInput:
                profileEpochSameInput(trainOpPack, numEpochs, X_data, Y_data)
            else:
                profileEpochDiffInput(trainOpPack, numEpochs, X_data, Y_data)
