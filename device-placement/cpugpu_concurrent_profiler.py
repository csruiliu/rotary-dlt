from __future__ import division
import numpy as np
from timeit import default_timer as timer
import os
import tensorflow as tf
import multiprocessing as mp

from dnn_model import DnnModel
from img_utils import *
import config as cfg_yml

import argparse

def generate_workload():
    cpu_job_queue = mp.Queue()
    gpu_job_queue = mp.Queue()

    workloadNum = sum(cpuModelNum) + sum(gpuModelNum)
    np.random.seed(randSeed)
    model_name_abbr = np.random.choice(randSeed, workloadNum, replace=False).tolist()

    for gidx, gnum in enumerate(gpuModelNum):
        for gpu_job in range(gnum):
            if not gpu_job_queue.full():
                gpu_job_queue.put([gpuModelType[gidx], model_name_abbr.pop(), gpuBatchSize[gidx], gpuOptimizer[gidx], gpuLearnRate[gidx], gpuActivation[gidx]])

    for cidx, cnum in enumerate(cpuModelNum):
        for cpu_job in range(cnum):
            if not cpu_job_queue.full():
                cpu_job_queue.put([cpuModelType[cidx], model_name_abbr.pop(), cpuBatchSize[cidx], cpuOptimizer[cidx], cpuLearnRate[cidx], cpuActivation[cidx]])

    return gpu_job_queue, cpu_job_queue


def generate_workload_from_cmd():
    cpu_job_queue = mp.Queue()
    gpu_job_queue = mp.Queue()

    workloadNum = cpuModelNum + gpuModelNum
    np.random.seed(randSeed)
    model_name_abbr = np.random.choice(randSeed, workloadNum, replace=False).tolist()

    for gpu_job in range(gpuModelNum):
        if not gpu_job_queue.full():
            gpu_job_queue.put([gpuModelType, model_name_abbr.pop(), batchSize, gpuOptimizer[0], gpuLearnRate[0], gpuActivation[0]])

    for cpu_job in range(cpuModelNum):
        if not cpu_job_queue.full():
            cpu_job_queue.put([cpuModelType, model_name_abbr.pop(), batchSize, cpuOptimizer[0], cpuLearnRate[0], cpuActivation[0]])

    return gpu_job_queue, cpu_job_queue


def run_single_job_gpu(model_type, model_instance, batch_size, optimizer, learning_rate, activation, assign_device):
    with tf.device(assign_device):
        features = tf.placeholder(tf.float32, [None, imgWidth, imgHeight, numChannels])
        labels = tf.placeholder(tf.int64, [None, numClasses])

        dm = DnnModel(model_type, str(model_instance), 1, imgHeight, imgWidth, numChannels, numClasses, batch_size, optimizer, learning_rate, activation, False)
        modelEntity = dm.getModelEntity()
        modelLogit = modelEntity.build(features)
        trainOps = modelEntity.train(modelLogit, labels)

        Y_data = load_imagenet_labels_onehot(label_path, numClasses)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        image_list = sorted(os.listdir(image_path_raw))

        step_time = 0
        step_count = 0

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            num_batch = Y_data.shape[0] // batch_size
            for i in range(num_batch):
                print('{}-{}-{} on gpu: step {} / {}'.format(model_type, batch_size, model_instance, i + 1, num_batch))
                if (i + 1) % recordMarker == 0:
                    start_time = timer()

                    batch_offset = i * batch_size
                    batch_end = (i + 1) * batch_size
                    batch_list = image_list[batch_offset:batch_end]
                    X_mini_batch_feed = load_imagenet_raw(image_path_raw, batch_list, imgHeight, imgWidth)
                    Y_mini_batch_feed = Y_data[batch_offset:batch_end, :]
                    sess.run(trainOps, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})

                    end_time = timer()
                    dur_time = end_time - start_time
                    print("step time:", dur_time)
                    step_time += dur_time
                    step_count += 1
                else:
                    batch_offset = i * batch_size
                    batch_end = (i + 1) * batch_size
                    batch_list = image_list[batch_offset:batch_end]
                    X_mini_batch_feed = load_imagenet_raw(image_path_raw, batch_list, imgHeight, imgWidth)
                    Y_mini_batch_feed = Y_data[batch_offset:batch_end, :]
                    sess.run(trainOps, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})

        print(step_time)
        print(step_count)
        print("average step time:", step_time / step_count * 1000)


def run_single_job_cpu(model_type, model_instance, batch_size, optimizer, learning_rate, activation, assign_device):
    with tf.device(assign_device):
        features = tf.placeholder(tf.float32, [None, imgWidth, imgHeight, numChannels])
        labels = tf.placeholder(tf.int64, [None, numClasses])

        dm = DnnModel(model_type, str(model_instance), 1, imgHeight, imgWidth, numChannels, numClasses, batch_size, optimizer, learning_rate, activation, False)
        modelEntity = dm.getModelEntity()
        modelLogit = modelEntity.build(features)
        trainOps = modelEntity.train(modelLogit, labels)

        Y_data = load_imagenet_labels_onehot(label_path, numClasses)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        image_list = sorted(os.listdir(image_path_raw))

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            num_batch = Y_data.shape[0] // batch_size
            for i in range(num_batch):
                print('{}-{}-{} on cpu: step {} / {}'.format(model_type, batch_size, model_instance, i + 1, num_batch))
                batch_offset = i * batch_size
                batch_end = (i + 1) * batch_size
                batch_list = image_list[batch_offset:batch_end]
                X_mini_batch_feed = load_imagenet_raw(image_path_raw, batch_list, imgHeight, imgWidth)
                Y_mini_batch_feed = Y_data[batch_offset:batch_end, :]
                sess.run(trainOps, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})


def consumer_gpu(queue, assign_device):
    if not queue.empty():
        gpu_job = queue.get()
        p = mp.Process(target=run_single_job_gpu, args=(gpu_job[0], gpu_job[1], gpu_job[2], gpu_job[3], gpu_job[4], gpu_job[5], assign_device))
        p.start()
        p.join()


def consumer_cpu(queue, assign_device):
    for _ in range(osThreadNum):
        if not queue.empty():
            cpu_job = queue.get()
            p = mp.Process(target=run_single_job_cpu, args=(cpu_job[0], cpu_job[1], cpu_job[2], cpu_job[3], cpu_job[4], cpu_job[5], assign_device))
            p.start()
        else:
            break


if __name__ == "__main__":

    ########################################
    # Get parameters using config
    ########################################

    randSeed = cfg_yml.rand_seed
    recordMarker = cfg_yml.record_marker
    useRawImage = cfg_yml.use_raw_image
    measureStep = cfg_yml.measure_step

    available_gpu_num = cfg_yml.available_gpu_num

    # cpuModelType = cfg_yml.cpu_model_type
    # cpuModelNum = cfg_yml.cpu_model_num
    # cpuBatchSize = cfg_yml.cpu_batch_size
    cpuLearnRate = cfg_yml.cpu_learning_rate
    cpuActivation = cfg_yml.cpu_activation
    cpuOptimizer = cfg_yml.cpu_optimizer

    # gpuModelType = cfg_yml.gpu_model_type
    # gpuModelNum = cfg_yml.gpu_model_num
    # gpuBatchSize = cfg_yml.gpu_batch_size
    gpuLearnRate = cfg_yml.gpu_learning_rate
    gpuActivation = cfg_yml.gpu_activation
    gpuOptimizer = cfg_yml.gpu_optimizer

    #########################################################################
    # Parameters read from command, but can be placed by read from config
    #########################################################################

    parser = argparse.ArgumentParser()
    parser.add_argument('-cm', '--cpu_model', required=True, action='store', type=str,
                        help='cpu model type [resnet,mobilenet,mlp,densenet]')
    parser.add_argument('-cn', '--cpu_model_num', required=True, action='store', type=int,
                        help='indicate the number of cpu model')
    parser.add_argument('-gm', '--gpu_model', required=True, action='store', type=str,
                        help='gpu model type [resnet,mobilenet,mlp,densenet]')
    parser.add_argument('-gn', '--gpu_model_num', required=True, action='store', type=int,
                        help='indicate the number of gpu model')
    parser.add_argument('-b', '--batch_size', required=True, action='store', type=int,
                        help='batch size [32,50,64,100,128]')
    parser.add_argument('-d', '--dataset', required=True, action='store', type=str,
                        help='training set [imagenet, cifar10]')

    args = parser.parse_args()

    cpuModelType = args.cpu_model
    cpuModelNum = args.cpu_model_num
    gpuModelType = args.gpu_model
    gpuModelNum = args.gpu_model_num
    batchSize = args.batch_size
    trainData = args.dataset

    ##########################
    # Build Workload
    ##########################

    osThreadNum = mp.cpu_count()
    training_gpu_queue, training_cpu_queue = generate_workload_from_cmd()

    ###########################################################
    # Build and train model due to input dataset
    ###########################################################

    if trainData == 'imagenet':
        image_path_raw = cfg_yml.imagenet_t1k_img_path
        image_path_bin = cfg_yml.imagenet_t1k_bin_path
        label_path = cfg_yml.imagenet_t1k_label_path

        imgWidth = cfg_yml.img_width_imagenet
        imgHeight = cfg_yml.img_height_imagenet
        numChannels = cfg_yml.num_channels_rgb
        numClasses = cfg_yml.num_class_imagenet

        features = tf.placeholder(tf.float32, [None, imgWidth, imgHeight, numChannels])
        labels = tf.placeholder(tf.int64, [None, numClasses])

    elif trainData == 'cifar10':
        imgWidth = cfg_yml.img_width_cifar10
        imgHeight = cfg_yml.img_height_cifar10
        numChannels = cfg_yml.num_channels_rgb
        numClasses = cfg_yml.num_class_cifar10

        features = tf.placeholder(tf.float32, [None, imgWidth, imgHeight, numChannels])
        labels = tf.placeholder(tf.int64, [None, numClasses])

    proc_gpu_list = list()

    for gn in range(available_gpu_num):
        assign_gpu = '/gpu:' + str(gn)
        device_proc_gpu = mp.Process(target=consumer_gpu, args=(training_gpu_queue, assign_gpu))
        proc_gpu_list.append(device_proc_gpu)

    for device_proc_gpu in proc_gpu_list:
        device_proc_gpu.start()

    device_proc_cpu = mp.Process(target=consumer_cpu, args=(training_cpu_queue, '/cpu:0'))
    device_proc_cpu.start()
