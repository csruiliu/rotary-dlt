from multiprocessing import Process, Manager, Value
import time
from timeit import default_timer as timer
import os
import operator as opr
import numpy as np
import tensorflow as tf

import config.config_parameter as cfg_para
from tools.workload_func import generate_workload_hyperparamsearch
from hypersched_trail import Trail, State


class State(Enum):
    RUN = 1
    PAUSE = 2
    STOP = 3


class Trail:
    def __init__(self, job_conf):
        self.trail_id = job_conf['job_id']
        self.model_type = job_conf['model_type']
        self.layer_number = job_conf['model_layer_num']
        self.batch_size = job_conf['batch_size']
        self.optimizer = job_conf['optimizer']
        self.learning_rate = job_conf['learning_rate']
        self.train_dataset = job_conf['train_dataset']

        # record how many training epochs has been launched
        self.train_progress = 0
        self.state = None

        # the updated accuracy
        self.cur_accuracy = 0

        #######################################
        # Prepare training dataset
        #######################################

        self.img_w, self.img_h, self.num_chn, self.num_class = load_dataset_para(self.train_dataset)
        self.train_feature, self.train_label = load_train_dataset(self.train_dataset)
        self.eval_feature, self.eval_label = load_eval_dataset(self.train_dataset)

    def train_simulate(self):
        time.sleep(np.random.randint(1, 4))
        self.train_progress += 1
        print('[process-{},trail-{}] training progress: {} epochs'.format(os.getpid(), self.trail_id,
                                                                          self.train_progress))

    def evaluate_simulate(self):
        self.cur_accuracy += np.random.uniform(0, 0.2)
        print('[process-{},trail-{}] after {} epochs, accuracy: {}'.format(os.getpid(), self.trail_id,
                                                                           self.train_progress, self.cur_accuracy))

    def train(self, assign_device):
        with tf.device(assign_device):
            feature_ph = tf.placeholder(tf.float32, [None, self.img_w, self.img_h, self.num_chn])
            label_ph = tf.placeholder(tf.int64, [None, self.num_class])

            train_model = ModelImporter(self.model_type, str(self.trail_id),
                                        self.layer_number, self.img_h,
                                        self.img_w, self.num_chn,
                                        self.num_class, self.batch_size,
                                        self.optimizer, self.learning_rate,
                                        activation='relu', batch_padding=False)

            model_entity = train_model.get_model_entity()
            model_logit = model_entity.build(feature_ph, is_training=True)
            model_train_op = model_entity.train(model_logit, label_ph)

            config = tf.ConfigProto()
            config.allow_soft_placement = True
            config.gpu_options.allow_growth = True

            if self.train_dataset == 'imagenet':
                train_data_list = sorted(os.listdir(self.train_feature))

            with tf.Session(config=config) as sess:
                num_batch = self.train_label.shape[0] // self.batch_size

                for i in range(num_batch):
                    # print('step {} / {}'.format(i + 1, num_batch))
                    batch_offset = i * self.batch_size
                    batch_end = (i + 1) * self.batch_size

                    if self.train_dataset == 'imagenet':
                        batch_list = train_data_list[batch_offset:batch_end]
                        train_data_batch = load_imagenet_raw(self.train_feature, batch_list, self.img_h, self.img_w)
                    else:
                        train_data_batch = self.train_feature[batch_offset:batch_end]

                    train_label_batch = self.train_label[batch_offset:batch_end]
                    sess.run(model_train_op, feed_dict={feature_ph: train_data_batch, label_ph: train_label_batch})

                self.train_progress += 1

    def evaluate(self, assign_device):
        with tf.device(assign_device):
            graph = tf.Graph()
            with graph.as_default():
                feature_ph = tf.placeholder(tf.float32, [None, self.img_w, self.img_h, self.num_chn])
                label_ph = tf.placeholder(tf.int64, [None, self.num_class])
                train_model = ModelImporter(self.model_type, str(self.trail_id),
                                            self.layer_number, self.img_h,
                                            self.img_w, self.num_chn,
                                            self.num_class, self.batch_size,
                                            self.optimizer, self.learning_rate,
                                            activation='relu', batch_padding=False)
                model_entity = train_model.get_model_entity()
                model_logit = model_entity.build(feature_ph, is_training=True)
                model_eval_op = model_entity.evaluate(model_logit, label_ph)

            config = tf.ConfigProto()
            config.allow_soft_placement = True
            config.gpu_options.allow_growth = True

            with tf.Session(graph=graph, config=config) as sess:
                if self.train_dataset == 'imagenet':
                    acc_sum = 0
                    num_eval_batch = self.train_label.shape[0] // 50
                    eval_data_list = sorted(os.listdir(self.eval_feature))
                    for n in range(num_eval_batch):
                        batch_offset = n * self.batch_size
                        batch_end = (n + 1) * self.batch_size
                        batch_eval_list = eval_data_list[batch_offset:batch_end]
                        feature_eval_batch = load_imagenet_raw(self.eval_feature,
                                                               batch_eval_list,
                                                               self.img_h,
                                                               self.img_w)
                        label_eval_batch = self.eval_label[batch_offset:batch_end]
                        acc_batch = sess.run(model_eval_op, feed_dict={feature_ph: feature_eval_batch,
                                                                       label_ph: label_eval_batch})
                        acc_sum += acc_batch
                else:
                    self.cur_accuracy = sess.run(model_eval_op, feed_dict={feature_ph: self.eval_feature,
                                                                           label_ph: self.eval_label})

    def set_state(self, arg_state):
        self.state = arg_state

    def get_state(self):
        return self.state

    def get_train_progress(self):
        return self.train_progress

    def get_accuracy(self):
        return self.cur_accuracy

    def get_trail_id(self):
        return self.trail_id


def get_available_trail(hp_workload_use, hp_workload_origin):
    next_trail = Trail(hp_workload_use.pop())
    if len(hp_workload_use) == 0:
        for _, value in enumerate(hp_workload_origin):
            hp_workload_use.append(value)
    return next_trail


def insert_sort_trail(trail_arg, live_trails_list):
    for trail_idx, trail_item in enumerate(live_trails_list):
        if trail_arg.get_accuracy() > trail_item.get_accuracy():
            live_trails_list.append(live_trails_list[-1])
            for idx in list(range(len(live_trails_list)-1, trail_idx, -1)):
                live_trails_list[idx] = live_trails_list[idx-1]
            live_trails_list[trail_idx] = trail_arg
            return
    live_trails_list.append(trail_arg)


def hypersched_schedule(live_trails_list,
                        hp_finish_flag,
                        hp_workload_use,
                        hp_workload_origin):
    # various epoch threshold according its original implementation
    MIN_EPOCH = 1
    MAX_EPOCH = 100

    # eta constant
    REDUCT_FACTOR = 4

    while True:
        try:
            trail = live_trails_list.pop()
            print('process {} started'.format(os.getpid()))
        except IndexError:
            live_trails_list.append(get_available_trail(hp_workload_use, hp_workload_origin))
        else:
            rung_check = 0
            while True:
                rung_epoch_threshold = np.power(REDUCT_FACTOR, rung_check) * MIN_EPOCH
                print('current rung epoch threshold {}'.format(rung_epoch_threshold))
                trail.set_state(State.RUN)
                trail.train_simulate()
                trail.evaluate_simulate()

                if trail.get_train_progress == MAX_EPOCH:
                    trail.set_state(State.STOP)
                    insert_sort_trail(trail, live_trails_list)
                    break

                if trail.get_train_progress() == rung_epoch_threshold:
                    print('trail {} reaches the epoch threshold'.format(trail.get_trail_id()))
                    trail.set_state(State.PAUSE)
                    live_trails_accuracy_list = list()
                    for st in sorted(live_trails_list, key=opr.attrgetter('cur_accuracy')):
                        live_trails_accuracy_list.append(st.get_accuracy())

                    pause_threshold = np.percentile(live_trails_accuracy_list, 1 / REDUCT_FACTOR)

                    if trail.get_accuracy() < pause_threshold:
                        trail.set_state(State.STOP)
                        insert_sort_trail(trail, live_trails_list)
                        break

                    trail.set_state(State.RUN)

                    rung_check += 1

            if hp_finish_flag.value == 1:
                print('process {} finished'.format(os.getpid()))
                break


def hypersched_timer(hp_deadline, hp_finish_flag):
    start_time = timer()
    while True:
        time.sleep(1)
        end_time = timer()
        dur_time = end_time - start_time
        if dur_time > hp_deadline:
            hp_finish_flag.value = 1
            print('timer process finished')
            break


def hypersched_run():
    job_num = cfg_para.hpsearch_job_num
    hpsearch_workload = generate_workload_hyperparamsearch(job_num)

    hpsearch_workload_use = Manager().list()
    for job in hpsearch_workload:
        hpsearch_workload_use.append(job)

    #######################################
    # Hyperparameter of HyperSched
    #######################################
    total_devices = cfg_para.sch_gpu_num
    slot_time_period = cfg_para.sch_slot_time_period
    slot_time_num = cfg_para.sch_time_slots_num

    # deadline for hyperparameter search (unit: second)
    hpsearch_deadline = slot_time_num * slot_time_period

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
        live_trails_list.append(get_available_trail(hpsearch_workload_use, hpsearch_workload))

    hpsearch_finish_flag = Value('i', 0)

    sch_proc_group = list()

    for _ in range(total_devices):
        sch_proc = Process(target=hypersched_schedule, args=(live_trails_list,
                                                             hpsearch_finish_flag,
                                                             hpsearch_workload_use,
                                                             hpsearch_workload))
        sch_proc_group.append(sch_proc)

    timer_proc = Process(target=hypersched_timer, args=(hpsearch_deadline, hpsearch_finish_flag))
    sch_proc_group.append(timer_proc)

    for p in sch_proc_group:
        p.start()

    for p in sch_proc_group:
        p.join()

    print('HyperSched is finished')
