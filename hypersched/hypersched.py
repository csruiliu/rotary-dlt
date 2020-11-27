from multiprocessing import Process, Manager, Value
import time
from timeit import default_timer as timer
import os
import numpy as np
import sys
import operator as opr
sys.path.append(os.path.abspath(".."))

from trail import Trail
from trail import State
import config.config_parameter as cfg_para
from utils.utils_workload_func import generate_workload_hyperparamsearch


def get_available_trail():
    next_trail = Trail(hpsearch_workload_use.pop())
    if len(hpsearch_workload_use) == 0:
        for idx, value in enumerate(hpsearch_workload):
            hpsearch_workload_use.append(value)
    return next_trail


def insert_sort_trail(trail_arg):
    for trail_idx, trail_item in enumerate(live_trails_list):
        if trail_arg.get_accuracy() > trail_item.get_accuracy():
            live_trails_list.append(live_trails_list[-1])
            for idx in list(range(len(live_trails_list)-1, trail_idx, -1)):
                live_trails_list[idx] = live_trails_list[idx-1]
            live_trails_list[trail_idx] = trail_arg
            return
    live_trails_list.append(trail_arg)


def hypersched_schedule():
    while True:
        try:
            trail = live_trails_list.pop()
            print('process {} started'.format(os.getpid()))
        except IndexError:
            live_trails_list.append(get_available_trail())
        else:
            train_count = 0
            while True:
                rung_epoch_threshold = np.power(min_epoch, train_count) * reduction_factor
                trail.set_state(State.RUN)
                trail.train_simulate()
                trail.evaluate_simulate()

                if trail.get_train_progress() == rung_epoch_threshold:
                    print('trail {} reaches the epoch threshold and is paused'.format(trail.get_trail_id()))
                    trail.set_state(State.PAUSE)
                    pause_threshold = np.percentile(sorted(live_trails_list, key=opr.attrgetter('cur_accuracy')),
                                                    1/reduction_factor)

                    if trail.get_accuracy() < pause_threshold:
                        trail.set_state(State.PAUSE)
                        insert_sort_trail(trail)
                        break

                train_count += 1

            if hpsearch_finish_flag.value == 1:
                print('process {} finished'.format(os.getpid()))
                break


def hypersched_timer():
    start_time = timer()

    while True:
        time.sleep(1)
        end_time = timer()
        dur_time = end_time - start_time
        if dur_time > hpsearch_deadline:
            hpsearch_finish_flag.value = 1
            print('timer process finished')
            break


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

    hpsearch_workload_use = Manager().list()
    for job in hpsearch_workload:
        hpsearch_workload_use.append(job)

    hpsearch_workload_size = len(hpsearch_workload)

    #######################################
    # Hyperparameter of HyperSched
    #######################################
    total_devices = cfg_para.sch_gpu_num
    slot_time_period = cfg_para.sch_slot_time_period
    slot_time_num = cfg_para.sch_time_slots_num

    # deadline for hyperparameter search (unit: second)
    hpsearch_deadline = slot_time_num * slot_time_period

    # various epoch threshold according its original implementation
    min_epoch = 1
    max_epoch = 100

    # eta constant
    reduction_factor = 4

    #######################################
    # Data Structure for HyperSched
    #######################################

    # a queue that can record the trails have been trained for at least one epoch
    live_trails_list = Manager().list()

    #######################################
    # HyperSched Starts
    #######################################

    # init the queue with some trails
    for _ in range(total_devices*2):
        live_trails_list.append(get_available_trail())

    hpsearch_finish_flag = Value('i', 0)

    sch_proc_group = list()

    for _ in range(total_devices):
        sch_proc = Process(target=hypersched_schedule)
        sch_proc_group.append(sch_proc)

    timer_proc = Process(target=hypersched_timer)
    sch_proc_group.append(timer_proc)

    for p in sch_proc_group:
        p.start()

    for p in sch_proc_group:
        p.join()

    print('HyperSched is finished')
