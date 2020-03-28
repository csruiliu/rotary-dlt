from __future__ import division
import numpy as np
from operator import itemgetter
from timeit import default_timer as timer
import os
import time
import tensorflow as tf
from multiprocessing import Process, Queue, Value

from dnn_model import DnnModel
from img_utils import *
import config as cfg_yml


def generate_workload():
    workload_list = list()
    np.random.seed(randSeed)
    model_name_abbr = np.random.choice(randSeed, workloadNum, replace=False).tolist()

    for idx, mt in enumerate(workloadModelType):
        workload_batchsize_index_list = np.random.choice(len(workloadBatchSize), workloadModelNum[idx], replace=False).tolist()
        workload_batchsize_list = list(itemgetter(*workload_batchsize_index_list)(workloadBatchSize))

        for bs in workload_batchsize_list:
            workload_list.append([mt, bs, model_name_abbr.pop()])

    return workload_list


def run_single_job_gpu(model_type, batch_size, model_instance, assign_device, proc_stop):
    with tf.device(assign_device):
        features = tf.placeholder(tf.float32, [None, imgWidth, imgHeight, numChannels])
        labels = tf.placeholder(tf.int64, [None, numClasses])

        dm = DnnModel(model_type, str(model_instance), 1, imgHeight, imgWidth, numChannels, numClasses, batch_size, 'Adam', 0.0001, 'relu', False)
        modelEntity = dm.getModelEntity()
        modelLogit = modelEntity.build(features)
        trainOps = modelEntity.train(modelLogit, labels)

        Y_data = load_imagenet_labels_onehot(label_path, numClasses)

        if not os.path.exists(ckpt_path):
            os.mkdir(ckpt_path)

        saver = tf.train.Saver(max_to_keep=1)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        image_list = sorted(os.listdir(image_path_raw))
        model_ckpt_path = ckpt_path + '/' + model_type + '_' + str(batch_size)

        with tf.Session(config=config) as sess:
            if os.path.exists(model_ckpt_path):
                saver.restore(sess, model_ckpt_path + '/' + model_type + '_' + str(batch_size))
            else:
                os.mkdir(model_ckpt_path)
                sess.run(tf.global_variables_initializer())

            num_batch = Y_data.shape[0] // batch_size
            for i in range(num_batch):
                if proc_stop.value == 1:
                    print("gpu training stop")
                    model_ckpt_save_path = model_ckpt_path + '/' + model_type + '_' + str(batch_size)
                    saver.save(sess, model_ckpt_save_path)
                    print("save the mode to path:", model_ckpt_save_path)
                    return

                print('{}-{}-{}: step {} / {}'.format(model_type, batch_size, model_instance, i + 1, num_batch))
                batch_offset = i * batch_size
                batch_end = (i + 1) * batch_size
                batch_list = image_list[batch_offset:batch_end]
                X_mini_batch_feed = load_imagenet_raw(image_path_raw, batch_list, imgHeight, imgWidth)
                Y_mini_batch_feed = Y_data[batch_offset:batch_end, :]
                sess.run(trainOps, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})

            model_ckpt_save_path = model_ckpt_path + '/' + model_type + '_' + str(batch_size)
            saver.save(sess, model_ckpt_save_path)
            print("save the mode to path:", model_ckpt_save_path)


def run_single_job_cpu(model_type, batch_size, model_instance, assign_device, proc_stop):
    with tf.device(assign_device):
        features = tf.placeholder(tf.float32, [None, imgWidth, imgHeight, numChannels])
        labels = tf.placeholder(tf.int64, [None, numClasses])

        dm = DnnModel(model_type, str(model_instance), 1, imgHeight, imgWidth, numChannels, numClasses, batch_size, 'Adam', 0.0001, 'relu', False)
        modelEntity = dm.getModelEntity()
        modelLogit = modelEntity.build(features)
        trainOps = modelEntity.train(modelLogit, labels)

        Y_data = load_imagenet_labels_onehot(label_path, numClasses)

        if not os.path.exists(ckpt_path):
            os.mkdir(ckpt_path)

        saver = tf.train.Saver(max_to_keep=1)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        image_list = sorted(os.listdir(image_path_raw))
        model_ckpt_path = ckpt_path + '/' + model_type + '_' + str(batch_size)

        with tf.Session(config=config) as sess:
            if os.path.exists(model_ckpt_path):
                saver.restore(sess, model_ckpt_path + '/' + model_type + '_' + str(batch_size))
            else:
                os.mkdir(model_ckpt_path)
                sess.run(tf.global_variables_initializer())

            num_batch = Y_data.shape[0] // batch_size
            for i in range(num_batch):
                if proc_stop.value == 1:
                    print("cpu training stop")
                    model_ckpt_save_path = model_ckpt_path + '/' + model_type + '_' + str(batch_size)
                    saver.save(sess, model_ckpt_save_path)
                    print("save the mode to path:", model_ckpt_save_path)
                    return

                print('{}-{}-{}: step {} / {}'.format(model_type, batch_size, model_instance, i + 1, num_batch))
                batch_offset = i * batch_size
                batch_end = (i + 1) * batch_size
                batch_list = image_list[batch_offset:batch_end]
                X_mini_batch_feed = load_imagenet_raw(image_path_raw, batch_list, imgHeight, imgWidth)
                Y_mini_batch_feed = Y_data[batch_offset:batch_end, :]
                sess.run(trainOps, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})

            model_ckpt_save_path = model_ckpt_path + '/' + model_type + '_' + str(batch_size)
            saver.save(sess, model_ckpt_save_path)
            print("save the mode to path:", model_ckpt_save_path)


def producer(queue, proc_stop):
    index = 0
    while True:
        if proc_stop.value == 1:
            print("producer stop")
            break
        if not queue.full():
            job_idx = index % workloadNum
            queue.put(workload_placement[job_idx])
            index += 1


def consumer_gpu(queue, proc_stop, assign_device):
    while True:
        if proc_stop.value == 1:
            print("consumer gpu stop")
            break
        if not queue.empty():
            job = queue.get()
            p = Process(target=run_single_job_gpu, args=(job[0], job[1], job[2], assign_device, proc_stop))
            p.start()
            p.join()


def consumer_cpu(queue, proc_stop, assign_device):
    while True:
        if proc_stop.value == 1:
            print("consumer cpu stop")
            break
        if not queue.empty():
            job = queue.get()
            p = Process(target=run_single_job_cpu, args=(job[0], job[1], job[2], assign_device, proc_stop))
            p.start()
            p.join()


if __name__ == "__main__":

    #########################
    # Parameters
    #########################

    imgWidth = cfg_yml.img_width
    imgHeight = cfg_yml.img_height
    numChannels = cfg_yml.num_channels
    numClasses = cfg_yml.num_classes
    randSeed = cfg_yml.rand_seed

    workloadModelType = cfg_yml.workload_model_type
    workloadModelNum = cfg_yml.workload_model_num
    workloadBatchSize = cfg_yml.workload_batch_size
    workloadActivation = cfg_yml.workload_activation
    workloadOptimizer = cfg_yml.workload_opt
    workloadLearnRate = cfg_yml.workload_learning_rate
    workloadNumLayer = cfg_yml.workload_num_layer

    deviceNum = cfg_yml.device_num
    totalEpoch = cfg_yml.total_epochs

    #########################
    # Build Workload
    #########################

    workloadNum = sum(workloadModelNum)
    workload_placement = generate_workload()

    #########################
    # Model Placement
    #########################

    image_path_raw = cfg_yml.imagenet_t10k_img_path
    image_path_bin = cfg_yml.imagenet_t10k_bin_path
    label_path = cfg_yml.imagenet_t10k_label_path
    ckpt_path = cfg_yml.ckpt_path

    test_image_path_raw = cfg_yml.imagenet_t1k_img_path
    test_image_path_bin = cfg_yml.imagenet_t1k_bin_path
    test_label_path = cfg_yml.imagenet_t1k_label_path
    robin_time_limit = cfg_yml.robin_time_limit
    available_cpu_num = cfg_yml.robin_available_cpu_num
    available_gpu_num = cfg_yml.robin_available_gpu_num

    training_job_queue = Queue(2)
    stop_flag = Value("i", 0)

    producer_proc = Process(target=producer, args=(training_job_queue, stop_flag))

    proc_cpu_list = list()
    proc_gpu_list = list()

    for cn in range(available_cpu_num):
        assign_cpu = '/cpu:'+str(cn)
        device_proc_cpu = Process(target=consumer_cpu, args=(training_job_queue, stop_flag, assign_cpu))
        proc_cpu_list.append(device_proc_cpu)

    for gn in range(available_gpu_num):
        assign_gpu = '/gpu:' + str(gn)
        device_proc_gpu = Process(target=consumer_gpu, args=(training_job_queue, stop_flag, assign_gpu))
        proc_gpu_list.append(device_proc_gpu)

    producer_proc.start()

    for device_proc_gpu in proc_gpu_list:
        device_proc_gpu.start()

    for device_proc_cpu in proc_cpu_list:
        device_proc_cpu.start()

    start_time = timer()
    while True:
        time.sleep(1)
        end_time = timer()
        dur_time = end_time - start_time
        if dur_time > robin_time_limit:
            stop_flag.value = 1
            break

    print("total running time:", dur_time)
