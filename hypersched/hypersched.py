import tensorflow as tf
import numpy as np
from multiprocessing import Process, Manager, Queue
from timeit import default_timer as timer
import os
import sys
sys.path.append(os.path.abspath(".."))

import config.config_parameter as cfg_para
import config.config_path as cfg_path
from utils.utils_workload_func import generate_workload_hyperparamsearch
from utils.utils_img_func import load_imagenet_labels_onehot, load_cifar10_keras, load_mnist_image, load_mnist_label_onehot

tf.compat.v1.enable_v2_behavior()


def get_new_job():
    next_job = hpsearch_workload_use.pop()
    if len(hpsearch_workload_use) == 0:
        for idx, value in enumerate(hpsearch_workload):
            hpsearch_workload_use.append(value)
    return next_job


def hypersched_run_simulate():
    start_time = timer()

    for i in range(total_atoms):
        live_jobs.put(get_new_job())

    while True:
        if live_jobs.empty():
            break
        else:
            for a_idx in range(total_atoms):
                cur_job = live_jobs.get()
                cur_job_idx = cur_job['job_id']
                # run the job for one epoch
                cur_job_accuracy = np.random.random()

        end_time = timer()
        dur_time = end_time - start_time
        if dur_time > hpsearch_deadline:
            break

    print('HyperSched is finished')


def hypersched_run():
    pass


if __name__ == "__main__":
    #######################################
    # Parameters of workload
    #######################################
    hpsearch_job_num = cfg_para.hpsearch_job_num
    hpsearch_model_type = cfg_para.hpsearch_model_type
    hpsearch_batch_size_set = cfg_para.hpsearch_batch_size_set
    hpsearch_layer_set = cfg_para.hpsearch_layer_set
    hpsearch_optimizer_set = cfg_para.hpsearch_optimizer_set
    hpsearch_learning_rate_set = cfg_para.hpsearch_learning_rate_set
    hpsearch_train_dataset = cfg_para.hpsearch_train_dataset

    hpsearch_workload = generate_workload_hyperparamsearch(hpsearch_job_num, hpsearch_model_type, hpsearch_layer_set,
                                                           hpsearch_batch_size_set, hpsearch_optimizer_set,
                                                           hpsearch_learning_rate_set, hpsearch_train_dataset)

    hpsearch_workload_use = hpsearch_workload.copy()

    hpsearch_workload_size = len(hpsearch_workload)

    #######################################
    # Hyperparameter of HyperSched
    #######################################
    total_atoms = cfg_para.sch_gpu_num
    slot_time_period = cfg_para.sch_slot_time_period
    slot_time_num = cfg_para.sch_time_slots_num

    # deadline for hyperparameter search (unit: second)
    hpsearch_deadline = slot_time_num * slot_time_period

    #######################################
    # HyperSched Starts
    #######################################

    # record the progress of each job and sort them in a list
    job_list = Manager().list([0] * hpsearch_workload_size)
    live_jobs = Queue()

    hypersched_run_simulate()

    #index_min = min(range(len(values)), key=values.__getitem__)



