from __future__ import division
import tensorflow as tf
from tensorflow.python.client import device_lib
import numpy as np
from operator import itemgetter
from timeit import default_timer as timer
from multiprocessing import Process
import os
import shutil

from dnn_model import DnnModel
from img_utils import *
import config as cfg_yml


def generate_workload():
    workload_dict = dict()

    np.random.seed(randSeed)

    for idx, mt in enumerate(workloadModelType):
        workload_batchsize_index_list = np.random.choice(len(workloadBatchSize), workloadModelNum[idx], replace=False).tolist()
        workload_batchsize_list = list(itemgetter(*workload_batchsize_index_list)(workloadBatchSize))
        workload_dict[mt] = workload_batchsize_list

    return workload_dict


def get_device_list():
    local_devices = device_lib.list_local_devices()
    return [x.name for x in local_devices if x.device_type == 'GPU' or x.device_type == 'CPU']


def robin_resource_allocation():
    workload_list = list()
    model_name_abbr = np.random.choice(randSeed, workloadNum, replace=False).tolist()
    for key, value in workload_placement.items():
        for v in value:
            temp = [key, v, model_name_abbr.pop()]
            workload_list.append(temp)

    start_time = timer()
    while True:
        for job in workload_list:
            p = Process(target=run_single_job, args=(job[0], job[1], job[2]))
            p.start()
            p.join()
            end_time = timer()
            dur_time = end_time - start_time
            if dur_time > robin_time_limit:
                print(workload_list)
                print('total running time:', dur_time)
                return


def run_single_job(model_type, batch_size, model_instance):
    features = tf.placeholder(tf.float32, [None, imgWidth, imgHeight, numChannels])
    labels = tf.placeholder(tf.int64, [None, numClasses])

    dm = DnnModel(model_type, str(model_instance), 1, imgHeight, imgWidth, numChannels, numClasses, batch_size, 'Adam', 0.0001, 'relu', False)
    modelEntity = dm.getModelEntity()
    modelLogit = modelEntity.build(features)
    trainOps = modelEntity.train(modelLogit, labels)

    Y_data = load_imagenet_labels_onehot(label_path, numClasses)

    if not os.path.exists(ckpt_path):
        os.mkdir(ckpt_path)

    saver = tf.train.Saver()

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
            print('step %d / %d' % (i + 1, num_batch))
            batch_offset = i * batch_size
            batch_end = (i + 1) * batch_size
            batch_list = image_list[batch_offset:batch_end]
            X_mini_batch_feed = load_imagenet_raw(image_path_raw, batch_list, imgHeight, imgWidth)
            Y_mini_batch_feed = Y_data[batch_offset:batch_end, :]
            sess.run(trainOps, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})

        if os.path.exists(model_ckpt_path):
            shutil.rmtree(model_ckpt_path)
            os.mkdir(model_ckpt_path)
        else:
            os.mkdir(model_ckpt_path)
        saver.save(sess, model_ckpt_path + '/' + model_type + '_' + str(batch_size))


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
    robin_resource_allocation()