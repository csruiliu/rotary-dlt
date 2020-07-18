#!/usr/bin/env python3

'''
Description:
Profiling the single training step of a GPU job when multiple jobs are running on CPU concurrently
'''

import os
import argparse
import multiprocessing as mp
from timeit import default_timer as timer
import numpy as np
import tensorflow as tf

import config_parameter as cfg_para_yml
import config_path as cfg_path_yml
from model_importer import ModelImporter
from utils_img_func import load_imagenet_labels_onehot, load_imagenet_bin, load_imagenet_raw, load_cifar_train

'''
def generate_workload_from_cfg():
    cpu_job_queue = mp.Queue()
    gpu_job_queue = mp.Queue()

    workload_num = sum(_cpu_model_num) + sum(_gpu_model_num)
    np.random.seed(_rand_seed)
    model_name_abbr = np.random.choice(_rand_seed, workload_num, replace=False).tolist()

    for gidx, gnum in enumerate(_gpu_model_num):
        for gpu_job in range(gnum):
            if not gpu_job_queue.full():
                gpu_job_queue.put([_gpu_model_type[gidx], model_name_abbr.pop(), _gpu_batch_size[gidx], _gpu_optimizer[gidx], _gpu_learn_rate[gidx], _gpu_activation[gidx]])

    for cidx, cnum in enumerate(_cpu_model_num):
        for cpu_job in range(cnum):
            if not cpu_job_queue.full():
                cpu_job_queue.put([_cpu_model_type[cidx], model_name_abbr.pop(), _cpu_batch_size[cidx], _cpu_optimizer[cidx], _cpu_learn_rate[cidx], _cpu_activation[cidx]])

    return gpu_job_queue, cpu_job_queue
'''


def generate_workload_from_cmd():
    cpu_job_queue = mp.Queue()
    gpu_job_queue = mp.Queue()

    workload_num = _cpu_model_num + _gpu_model_num
    np.random.seed(_rand_seed)
    model_name_abbr = np.random.choice(_rand_seed, workload_num, replace=False).tolist()

    for _ in range(_gpu_model_num):
        if not gpu_job_queue.full():
            gpu_job_queue.put([_gpu_model_type, model_name_abbr.pop(), _train_batch_size, _gpu_optimizer[0], _gpu_learn_rate[0], _gpu_activation[0]])

    for _ in range(_cpu_model_num):
        if not cpu_job_queue.full():
            cpu_job_queue.put([_cpu_model_type, model_name_abbr.pop(), _train_batch_size, _cpu_optimizer[0], _cpu_learn_rate[0], _cpu_activation[0]])

    return gpu_job_queue, cpu_job_queue


def run_single_job_gpu(model_type, model_instance, batch_size, optimizer, learning_rate, activation, assign_device):
    with tf.device(assign_device):
        features = tf.placeholder(tf.float32, [None, _img_width, _img_height, _num_channels])
        labels = tf.placeholder(tf.int64, [None, _num_classes])

        train_model = ModelImporter(model_type, str(model_instance), 1, _img_height, _img_width, _num_channels, _num_classes,
                               batch_size, optimizer, learning_rate, activation, False)
        model_entity = train_model.get_model_entity()
        model_logit = model_entity.build(features)
        train_ops = model_entity.train(model_logit, labels)

        if _use_raw_image:
            image_list = sorted(os.listdir(_image_path_raw))

        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        tf_config.allow_soft_placement = True

        step_time = 0
        step_count = 0

        print("gpu job start...")

        with tf.Session(config=tf_config) as sess:
            sess.run(tf.global_variables_initializer())
            num_batch = _Y_data.shape[0] // batch_size
            for i in range(num_batch):
                print('**GPU JOB**: {}-{}-{} on gpu [{}]: step {} / {}'.format(model_type, batch_size, model_instance, timer(), i + 1, num_batch))
                if (i + 1) % _record_marker == 0:
                    start_time = timer()
                    batch_offset = i * batch_size
                    batch_end = (i + 1) * batch_size
                    if _use_raw_image:
                        batch_list = image_list[batch_offset:batch_end]
                        X_mini_batch_feed = load_imagenet_raw(_image_path_raw, batch_list, _img_height, _img_width)
                    else:
                        X_mini_batch_feed = _X_data[batch_offset:batch_end, :, :, :]
                    Y_mini_batch_feed = _Y_data[batch_offset:batch_end, :]
                    sess.run(train_ops, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})
                    end_time = timer()
                    dur_time = end_time - start_time
                    print("step time:", dur_time)
                    step_time += dur_time
                    step_count += 1
                else:
                    batch_offset = i * batch_size
                    batch_end = (i + 1) * batch_size
                    if _use_raw_image:
                        batch_list = image_list[batch_offset:batch_end]
                        X_mini_batch_feed = load_imagenet_raw(_image_path_raw, batch_list, _img_height, _img_width)
                    else:
                        X_mini_batch_feed = _X_data[batch_offset:batch_end, :, :, :]
                    Y_mini_batch_feed = _Y_data[batch_offset:batch_end, :]
                    sess.run(train_ops, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})

        print(step_time)
        print(step_count)
        print('GPU job average step time [{}]: {}'.format(timer(), step_time / step_count * 1000))


def run_single_job_cpu(model_type, model_instance, batch_size, optimizer, learning_rate, activation, assign_device, proc_idx=0):
    with tf.device(assign_device):
        features = tf.placeholder(tf.float32, [None, _img_width, _img_height, _num_channels])
        labels = tf.placeholder(tf.int64, [None, _num_classes])

        train_model = ModelImporter(model_type, str(model_instance), 1, _img_height, _img_width, _num_channels, _num_classes,
                               batch_size, optimizer, learning_rate, activation, False)
        model_entity = train_model.get_model_entity()
        model_logit = model_entity.build(features)
        train_ops = model_entity.train(model_logit, labels)

        if _use_raw_image:
            image_list = sorted(os.listdir(_image_path_raw))

        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        tf_config.allow_soft_placement = True

        print("cpu job start...")

        step_time = 0
        step_count = 0

        with tf.Session(config=tf_config) as sess:
            sess.run(tf.global_variables_initializer())
            num_batch = _Y_data.shape[0] // batch_size
            for i in range(num_batch):
                print('**CPU JOB**: Proc-{}, {}-{}-{} on cpu [{}]: step {} / {}'.format(proc_idx, model_type, batch_size, model_instance, timer(), i + 1, num_batch))
                if (i + 1) % _record_marker == 0:
                    start_time = timer()
                    batch_offset = i * batch_size
                    batch_end = (i + 1) * batch_size
                    if _use_raw_image:
                        batch_list = image_list[batch_offset:batch_end]
                        X_mini_batch_feed = load_imagenet_raw(_image_path_raw, batch_list, _img_height, _img_width)
                    else:
                        X_mini_batch_feed = _X_data[batch_offset:batch_end, :, :, :]
                    Y_mini_batch_feed = _Y_data[batch_offset:batch_end, :]
                    sess.run(train_ops, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})
                    end_time = timer()
                    dur_time = end_time - start_time
                    print("step time:", dur_time)
                    step_time += dur_time
                    step_count += 1
                else:
                    batch_offset = i * batch_size
                    batch_end = (i + 1) * batch_size
                    if _use_raw_image:
                        batch_list = image_list[batch_offset:batch_end]
                        X_mini_batch_feed = load_imagenet_raw(_image_path_raw, batch_list, _img_height, _img_width)
                    else:
                        X_mini_batch_feed = _X_data[batch_offset:batch_end, :, :, :]
                    Y_mini_batch_feed = _Y_data[batch_offset:batch_end, :]
                    sess.run(train_ops, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})

            print(step_time)
            print(step_count)
            print('CPU job average step time [{}]: {}'.format(timer(), step_time / step_count * 1000))


def consumer_gpu(queue, assign_device):
    if not queue.empty():
        gpu_job = queue.get()
        p = mp.Process(target=run_single_job_gpu, args=(gpu_job[0], gpu_job[1], gpu_job[2], gpu_job[3], gpu_job[4], gpu_job[5], assign_device))
        p.start()
        p.join()


def consumer_cpu(queue, assign_device):
    for procidx in range(osThreadNum):
        if not queue.empty():
            cpu_job = queue.get()
            p = mp.Process(target=run_single_job_cpu, args=(cpu_job[0], cpu_job[1], cpu_job[2], cpu_job[3], cpu_job[4], cpu_job[5], assign_device, procidx))
            p.start()
        else:
            break


if __name__ == "__main__":

    ########################################
    # Get parameters using config
    ########################################

    _rand_seed = cfg_para_yml.rand_seed
    _record_marker = cfg_para_yml.record_marker
    _use_raw_image = cfg_para_yml.use_raw_image
    _use_measure_step = cfg_para_yml.measure_step

    _available_gpu_num = cfg_para_yml.available_gpu_num

    # cpu_model_type = cfg_para_yml.cpu_model_type
    # cpu_model_num = cfg_para_yml.cpu_model_num
    # cpu_batch_size = cfg_para_yml.cpu_batch_size
    _cpu_learn_rate = cfg_para_yml.cpu_learning_rate
    _cpu_activation = cfg_para_yml.cpu_activation
    _cpu_optimizer = cfg_para_yml.cpu_optimizer

    # gpu_model_type = cfg_para_yml.gpu_model_type
    # gpu_model_num = cfg_para_yml.gpu_model_num
    # gpu_batch_size = cfg_para_yml.gpu_batch_size
    _gpu_learn_rate = cfg_para_yml.gpu_learning_rate
    _gpu_activation = cfg_para_yml.gpu_activation
    _gpu_optimizer = cfg_para_yml.gpu_optimizer

    #########################################################################
    # Parameters read from command, but can be placed by read from config
    #########################################################################

    parser = argparse.ArgumentParser()
    parser.add_argument('-cm', '--cpu_model', required=True, action='store',
                        choices=['resnet', 'mobilenet', 'densenet', 'mlp', 'scn'],
                        help='cpu model type [resnet,mobilenet,mlp,densenet]')
    parser.add_argument('-cn', '--cpu_model_num', required=True, action='store', type=int,
                        help='indicate the number of cpu model')
    parser.add_argument('-gm', '--gpu_model', required=True, action='store',
                        choices=['resnet', 'mobilenet', 'densenet', 'mlp', 'scn'],
                        help='gpu model type [resnet,mobilenet,mlp,densenet]')
    parser.add_argument('-gn', '--gpu_model_num', required=True, action='store', type=int,
                        help='indicate the number of gpu model')
    parser.add_argument('-b', '--batch_size', required=True, action='store', type=int,
                        help='batch size [32,50,64,100,128]')
    parser.add_argument('-d', '--dataset', required=True, action='store', choices=['imagenet', 'cifar10'],
                        help='training set [imagenet, cifar10]')

    args = parser.parse_args()

    _cpu_model_type = args.cpu_model
    _cpu_model_num = args.cpu_model_num
    _gpu_model_type = args.gpu_model
    _gpu_model_num = args.gpu_model_num
    _train_batch_size = args.batch_size
    _train_data = args.dataset

    ##########################
    # Build Workload
    ##########################

    osThreadNum = mp.cpu_count()
    training_gpu_queue, training_cpu_queue = generate_workload_from_cmd()

    ###########################################################
    # Build and train model due to input dataset
    ###########################################################

    if _train_data == 'imagenet':
        _image_path_raw = cfg_path_yml.imagenet_t1k_img_path
        _image_path_bin = cfg_path_yml.imagenet_t1k_bin_path
        _label_path = cfg_path_yml.imagenet_t1k_label_path

        _img_width = cfg_para_yml.img_width_imagenet
        _img_height = cfg_para_yml.img_height_imagenet
        _num_channels = cfg_para_yml.num_channels_rgb
        _num_classes = cfg_para_yml.num_class_imagenet

        if _use_raw_image:
            _Y_data = load_imagenet_labels_onehot(_label_path, _num_classes)
        else:
            _X_data = load_imagenet_bin(_image_path_bin, _num_channels, _img_width, _img_height)
            _Y_data = load_imagenet_labels_onehot(_label_path, _num_classes)

    elif _train_data == 'cifar10':
        _img_width = cfg_para_yml.img_width_cifar10
        _img_height = cfg_para_yml.img_height_cifar10
        _num_channels = cfg_para_yml.num_channels_rgb
        _num_classes = cfg_para_yml.num_class_cifar10

        _use_raw_image = False

        _cifar10_path = cfg_path_yml.cifar_10_path
        _X_data, _Y_data = load_cifar_train(_cifar10_path, _rand_seed)

    proc_gpu_list = list()

    for gn in range(_available_gpu_num):
        assign_gpu = '/gpu:' + str(gn)
        device_proc_gpu = mp.Process(target=consumer_gpu, args=(training_gpu_queue, assign_gpu))
        proc_gpu_list.append(device_proc_gpu)

    for device_proc_gpu in proc_gpu_list:
        device_proc_gpu.start()

    device_proc_cpu = mp.Process(target=consumer_cpu, args=(training_cpu_queue, '/cpu:0'))
    device_proc_cpu.start()
