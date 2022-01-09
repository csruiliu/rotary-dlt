import os
import queue
from datetime import datetime
import tensorflow as tf
import multiprocessing as mp
from time import perf_counter

from rotary.common.constants import JobSLO, JobStatus, SchedType
from rotary.common.property_utils import PropertyUtils
from rotary.common.log_utils import log_get_job, log_train_job, log_eval_job, log_time_accuracy
from rotary.common.model_utils import build_cv_model, build_nlp_model
from rotary.common.sched_utils import (init_tf_config,
                                       init_tf_vars,
                                       compared_item,
                                       get_bert_dataset,
                                       prepare_bert_dataset)

import rotary.reader.udtb_reader as udtb_reader
import rotary.reader.cifar_reader as cifar_reader


class SRF:
    ###############################
    # Shortest Runtime First
    ###############################
    def __init__(self,
                 path_file,
                 para_file,
                 knowledgebase_path,
                 ml_workload):

        # path config
        self.path_file = path_file
        self.para_file = para_file
        self.knowledgebase_path = knowledgebase_path

        path_cfg = PropertyUtils.load_property_file(self.path_file)
        para_cfg = PropertyUtils.load_property_file(self.para_file)

        # create folder if not exist
        self.ckpt_save_path = path_cfg['ckpt_save_path']
        if not os.path.exists(self.ckpt_save_path):
            os.makedirs(self.ckpt_save_path)

        #######################################################
        # get parameters from configuration
        #######################################################

        dlt_workload_cfg = para_cfg['dlt_workload']
        self.dlt_workload_size = dlt_workload_cfg['workload_size']
        self.dlt_residual_ratio = dlt_workload_cfg['residual_ratio']
        self.dlt_mobile_ratio = dlt_workload_cfg['mobile_ratio']
        self.dlt_lstm_ratio = dlt_workload_cfg['lstm_ratio']
        self.dlt_bert_ratio = dlt_workload_cfg['bert_ratio']
        self.dlt_others_ratio = dlt_workload_cfg['others_ratio']

        objective_cfg = para_cfg['objective']
        self.convergence_ratio = objective_cfg['convergence_ratio']
        self.accuracy_ratio = objective_cfg['accuracy_ratio']
        self.runtime_ratio = objective_cfg['runtime_ratio']

        self.random_seed = para_cfg['random_seed']
        self.num_gpu = para_cfg['num_gpu']
        self.running_slot = para_cfg['running_slot']

        #######################################################
        # prepare everything necessary
        #######################################################

        # init seaphore
        self.sem_runtime = mp.Semaphore(self.num_gpu)
        self.sem_others = mp.Semaphore(self.num_gpu)
        self.gpu_slot_runtime = mp.Array('i', [0] * self.num_gpu)
        self.gpu_slot_others = mp.Array('i', [0] * self.num_gpu)

        self.start_time_overall = perf_counter()

        # list with runtime-slo jobs only
        self.ml_workload_runtime = mp.Manager().list()

        # list with other slo jobs
        self.ml_workload_others = mp.Manager().list()
        # queue for checking if all jobs are completed
        self.job_queue_others = mp.Manager().Queue()

        # dict for storing job's current accuracy
        self.job_dict_accuracy = mp.Manager().dict()
        # dict for storing job's overall running time
        self.job_dict_runtime = mp.Manager().dict()
        # dict for storing job's overall training epochs
        self.job_dict_epoch = mp.Manager().dict()
        # dict for storing job's completion time (achieving SLO or being terminated)
        self.job_dict_completion_time = mp.Manager().dict()
        # dict for storing if the job achieving the SLO
        self.job_dict_attainment = mp.Manager().dict()

        # dict for tracking epochs with wall-time
        self.job_history_runtime = dict()
        # dict for tracking accuracy with wall-time
        self.job_history_accuracy = dict()

        self.proc_pool_runtime = mp.Pool(self.num_gpu, maxtasksperchild=1)
        self.proc_pool_others = mp.Pool(self.num_gpu, maxtasksperchild=1)

        #######################################################
        # start processing the workload
        #######################################################

        proc_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print('============= the whole exp starts at: {}===================='.format(proc_start_time))

        # init dicts
        for job in ml_workload:
            if job['goal_type'] == 'runtime':
                self.ml_workload_runtime.append(job)
            else:
                self.ml_workload_others.append(job)
                self.job_queue_others.put(job)

            job_key = str(job['id']) + '-' + job['model']

            self.job_dict_accuracy[job_key] = 0
            self.job_dict_runtime[job_key] = 0
            self.job_dict_epoch[job_key] = 0
            self.job_dict_completion_time[job_key] = 0
            self.job_dict_attainment[job_key] = 0

            init_sub_accuracy_list = mp.Manager().list()
            init_sub_accuracy_list.append('0:' + proc_start_time)
            self.job_history_accuracy[job_key] = init_sub_accuracy_list
            init_sub_runtime_list = mp.Manager().list()
            init_sub_runtime_list.append('0:' + proc_start_time)
            self.job_history_runtime[job_key] = init_sub_runtime_list

    def complete_job_runtime(self, job_name, gpu_device, attain):
        end_time_overall = perf_counter()
        self.job_dict_completion_time[job_name] = end_time_overall - self.start_time_overall
        self.job_dict_attainment[job_name] = attain
        self.gpu_slot_runtime[gpu_device] = 0
        self.sem_runtime.release()

    def complete_job_others(self, job_name, gpu_device, attain):
        end_time_overall = perf_counter()
        self.job_dict_completion_time[job_name] = end_time_overall - self.start_time_overall
        self.job_dict_attainment[job_name] = attain
        self.gpu_slot_others[gpu_device] = 0
        self.sem_others.release()

    def check_job_progress_runtime(self,
                                   job_name,
                                   cur_accuracy,
                                   job_slo_value,
                                   gpu_device):

        log_time_accuracy(job_name,
                          cur_accuracy,
                          self.job_history_runtime,
                          self.job_dict_epoch,
                          self.job_history_accuracy)

        if self.job_dict_epoch[job_name] >= job_slo_value:
            self.complete_job_runtime(job_name, gpu_device, attain=1)
            return JobStatus.COMPLETE_ATTAIN
        else:
            return JobStatus.INCOMPLETE

    def check_job_progress_others(self,
                                  job_name,
                                  job_slo,
                                  job_slo_value,
                                  job_slo_max_time,
                                  pre_accuracy,
                                  cur_accuracy,
                                  gpu_device):

        log_time_accuracy(job_name,
                          cur_accuracy,
                          self.job_history_runtime,
                          self.job_dict_epoch,
                          self.job_history_accuracy)

        if job_slo == 'accuracy':
            if self.job_dict_accuracy[job_name] >= job_slo_value:
                self.complete_job_others(job_name, gpu_device, attain=1)
                return JobStatus.COMPLETE_ATTAIN
            elif self.job_dict_epoch[job_name] >= job_slo_max_time:
                self.complete_job_others(job_name, gpu_device, attain=0)
                return JobStatus.COMPLETE_UNATTAIN
            else:
                return JobStatus.INCOMPLETE

        elif job_slo == 'convergence':
            delta = round(abs(cur_accuracy - pre_accuracy), 4)
            if delta <= job_slo_value:
                self.complete_job_runtime(job_name, gpu_device, attain=1)
                return JobStatus.COMPLETE_ATTAIN
            elif self.job_dict_epoch[job_name] >= job_slo_max_time:
                self.complete_job_others(job_name, gpu_device, attain=0)
                return JobStatus.COMPLETE_UNATTAIN
            else:
                return JobStatus.INCOMPLETE
        else:
            raise ValueError('the job objective type is not supported')

    def train_bert_model(self,
                         job_data,
                         gpu_device,
                         process_start_marker,
                         mode):
        # init the time counter for a slot
        running_slot_time = 0

        job_slo = job_data['goal_type']
        job_slo_value = job_data['goal_value']
        if job_slo == JobSLO.RUNTIME:
            job_slo_max_time = -1
        else:
            job_slo_max_time = job_data['goal_value_extra']

        job_id = job_data['id']
        job_model = job_data['model']
        job_name = str(job_id) + '-' + job_model
        assign_device = '/gpu:' + str(gpu_device)

        # create the folder for saving models
        model_ckpt_save_path = self.ckpt_save_path + '/' + job_name
        if not os.path.exists(model_ckpt_save_path):
            os.makedirs(model_ckpt_save_path)

        log_get_job(job_name, os.getpid(), assign_device)

        # Params for bert model and tokenization
        bert_path = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
        max_seq_length = 128
        offset = 500

        train_text, train_label = get_bert_dataset(max_seq_length)

        try:
            # start processing on the assigned device
            with tf.device(assign_device):
                with tf.Session(config=init_tf_config()) as sess:
                    (
                        train_input_ids,
                        train_input_masks,
                        train_segment_ids,
                        train_labels
                    ) = prepare_bert_dataset(bert_path, sess, train_text, train_label, max_seq_length)

                    model = build_nlp_model(model_type=job_model,
                                            max_length=128,
                                            opt=job_data['opt'],
                                            lr=job_data['learn_rate'])
                    logit, _ = model.build()

                    # load the checkpoint if it exists
                    if os.path.exists(model_ckpt_save_path + '/' + job_name + '.h5'):
                        logit.load_weights(model_ckpt_save_path + '/' + job_name + '.h5')

                    # Instantiate variables
                    init_tf_vars(sess)

                    # add the prepare time for this process
                    preparation_end_marker = perf_counter()
                    self.job_dict_runtime[job_name] += preparation_end_marker - preparation_end_marker

                    # check if the total runtime is less than running_slot
                    while running_slot_time < self.running_slot:
                        epoch_start_marker = perf_counter()
                        log_train_job(job_name, os.getpid(), assign_device)
                        logit.fit([train_input_ids, train_input_masks, train_segment_ids],
                                  train_labels,
                                  epochs=1,
                                  batch_size=job_data['batch_size'],
                                  verbose=0)

                        # start evaluation phrase
                        log_eval_job(job_name, os.getpid(), assign_device)
                        scores = logit.evaluate([train_input_ids[0:offset],
                                                 train_input_masks[0:offset],
                                                 train_segment_ids[0:offset]],
                                                train_labels[0:offset],
                                                verbose=0)
                        cur_accuracy = scores[1]

                        pre_accuracy = self.job_dict_accuracy[job_name]

                        epoch_end_marker = perf_counter()

                        # tracking the time and accuracy
                        self.job_dict_runtime[job_name] += epoch_end_marker - epoch_start_marker
                        self.job_dict_epoch[job_name] += 1
                        self.job_dict_accuracy[job_name] = cur_accuracy

                        if mode == SchedType.SCHED_RUNTIME:
                            job_progress = self.check_job_progress_runtime(job_name,
                                                                           cur_accuracy,
                                                                           job_slo_value,
                                                                           gpu_device)

                        else:
                            job_progress = self.check_job_progress_others(job_name,
                                                                          job_slo,
                                                                          job_slo_value,
                                                                          job_slo_max_time,
                                                                          pre_accuracy,
                                                                          cur_accuracy,
                                                                          gpu_device)

                        if job_progress == JobStatus.COMPLETE_ATTAIN:
                            logit.save(model_ckpt_save_path + '/' + job_name + '.h5')
                            msg = 'job {} reaches SLO'.format(job_id)
                            return msg
                        elif job_progress == JobStatus.COMPLETE_UNATTAIN:
                            logit.save(model_ckpt_save_path + '/' + job_name + '.h5')
                            msg = 'job {} is finished'.format(job_id)
                            return msg
                        else:
                            pass

                        if mode == SchedType.SCHED_OTHERS:
                            slot_end_marker = perf_counter()
                            running_slot_time = slot_end_marker - process_start_marker

                    # save the model/job since the job has run for the current slot but doesn't achieve SLO
                    logit.save(model_ckpt_save_path + '/' + job_name + '.h5')
        except RuntimeError:
            print('######### Unknown Error: {} #########'.format(job_name))
            self.gpu_slot_others[gpu_device] = 0
            self.sem_others.release()

    def train_lstm_model(self,
                         job_data,
                         gpu_device,
                         process_start_marker,
                         mode):
        # init the time counter for a slot
        running_slot_time = 0

        job_id = job_data['id']
        job_model = job_data['model']
        job_name = str(job_id) + '-' + job_model
        assign_device = '/gpu:' + str(gpu_device)

        # create the folder for saving models
        model_ckpt_save_path = self.ckpt_save_path + '/' + job_name
        if not os.path.exists(model_ckpt_save_path):
            os.makedirs(model_ckpt_save_path)

        log_get_job(job_name, os.getpid(), assign_device)

        job_slo = job_data['goal_type']
        job_slo_value = job_data['goal_value']
        if job_slo == JobSLO.RUNTIME:
            job_slo_max_time = -1
        else:
            job_slo_max_time = job_data['goal_value_extra']

        # load udtb dataset
        (train_sentences_x,
         val_sentences_x,
         train_tags_y,
         val_tags_y,
         MAX_LENGTH,
         word2index,
         tag2index) = udtb_reader.load_udtb_dataset()

        # start processing on the assigned device
        with tf.device(assign_device):
            # build model
            model = build_nlp_model(model_type=job_model,
                                    max_length=MAX_LENGTH,
                                    opt=job_data['opt'],
                                    lr=job_data['learn_rate'])

            logit, total_parameters = model.build(word2index, tag2index)

            # load the checkpoint if it exists
            if os.path.exists(model_ckpt_save_path + '/' + job_name + '.h5'):
                logit.load_weights(model_ckpt_save_path + '/' + job_name + '.h5')

            with tf.Session(config=init_tf_config()) as sess:
                sess.run(tf.global_variables_initializer())

                # add the preparation time for this process
                preparation_end_marker = perf_counter()
                self.job_dict_runtime[job_name] += preparation_end_marker - process_start_marker

                # check if the total runtime is less than running_slot
                while running_slot_time < self.running_slot:
                    epoch_start_marker = perf_counter()
                    log_train_job(job_name, os.getpid(), assign_device)
                    logit.fit(train_sentences_x,
                              udtb_reader.to_categorical(train_tags_y, len(tag2index)),
                              batch_size=job_data['batch_size'],
                              epochs=1,
                              verbose=0)

                    # start evaluation phrase
                    log_eval_job(job_name, os.getpid(), assign_device)
                    scores = logit.evaluate(val_sentences_x,
                                            udtb_reader.to_categorical(val_tags_y, len(tag2index)),
                                            verbose=0)
                    cur_accuracy = scores[1]

                    pre_accuracy = self.job_dict_accuracy[job_name]

                    epoch_end_marker = perf_counter()

                    # tracking the time and accuracy
                    self.job_dict_runtime[job_name] += epoch_end_marker - epoch_start_marker
                    self.job_dict_epoch[job_name] += 1
                    self.job_dict_accuracy[job_name] = cur_accuracy

                    if mode == SchedType.SCHED_RUNTIME:
                        job_progress = self.check_job_progress_runtime(job_name,
                                                                       cur_accuracy,
                                                                       job_slo_value,
                                                                       gpu_device)
                    else:
                        job_progress = self.check_job_progress_others(job_name,
                                                                      job_slo,
                                                                      job_slo_value,
                                                                      job_slo_max_time,
                                                                      pre_accuracy,
                                                                      cur_accuracy,
                                                                      gpu_device)

                    if job_progress == JobStatus.COMPLETE_ATTAIN:
                        logit.save(model_ckpt_save_path + '/' + job_name + '.h5')
                        msg = 'job {} reaches SLO'.format(job_id)
                        return msg
                    elif job_progress == JobStatus.COMPLETE_UNATTAIN:
                        logit.save(model_ckpt_save_path + '/' + job_name + '.h5')
                        msg = 'job {} is finished'.format(job_id)
                        return msg
                    else:
                        pass

                    if mode == SchedType.SCHED_OTHERS:
                        slot_end_marker = perf_counter()
                        running_slot_time = slot_end_marker - process_start_marker

                # save the model/job since the job has run for the current slot but doesn't achieve SLO
                logit.save(model_ckpt_save_path + '/' + job_name + '.h5')

    def train_cv_model(self,
                       job_data,
                       gpu_device,
                       process_start_marker,
                       mode):
        # init the time counter for a slot
        running_slot_time = 0

        job_id = job_data['id']
        job_model = job_data['model']
        job_name = str(job_id) + '-' + job_model
        assign_device = '/gpu:' + str(gpu_device)

        # create the folder for saving models
        model_ckpt_save_path = self.ckpt_save_path + '/' + job_name
        if not os.path.exists(model_ckpt_save_path):
            os.makedirs(model_ckpt_save_path)

        log_get_job(job_name, os.getpid(), assign_device)

        job_slo = job_data['goal_type']
        job_slo_value = job_data['goal_value']
        if job_slo == JobSLO.RUNTIME:
            job_slo_max_time = -1
        else:
            job_slo_max_time = job_data['goal_value_extra']

        train_batchsize = job_data['batch_size']

        # load cifar10 data
        (
            train_feature,
            train_labels,
            eval_feature,
            eval_labels
        ) = cifar_reader.load_cifar10_keras()

        try:
            with tf.device(assign_device):
                feature_ph = tf.placeholder(tf.float32, [None, 32, 32, 3])
                label_ph = tf.placeholder(tf.int64, [None, 10])
                train_op, eval_op, total_parameters = build_cv_model(job_data,
                                                                     n_class=10,
                                                                     feature=feature_ph,
                                                                     label=label_ph)

                # init the tf saver for checkpoint
                saver = tf.train.Saver()

                with tf.Session(config=init_tf_config()) as sess:
                    # check if the checkpoint file exist
                    checkpoint_file = model_ckpt_save_path + '/model_ckpt'
                    if os.path.isfile(checkpoint_file + '.meta'):
                        saver.restore(sess, checkpoint_file)
                    else:
                        sess.run(tf.global_variables_initializer())

                    num_batch = train_labels.shape[0] // train_batchsize

                    # add the preparation time for this process
                    preparation_end_marker = perf_counter()
                    self.job_dict_runtime[job_name] += preparation_end_marker - process_start_marker

                    # check if the total runtime is less than running_slot
                    while running_slot_time < self.running_slot:
                        epoch_start_marker = perf_counter()
                        log_train_job(job_name, os.getpid(), assign_device)
                        for b in range(num_batch):
                            batch_offset = b * train_batchsize
                            batch_end = (b + 1) * train_batchsize

                            train_data_batch = train_feature[batch_offset:batch_end]
                            train_label_batch = train_labels[batch_offset:batch_end]

                            sess.run(train_op, feed_dict={feature_ph: train_data_batch,
                                                          label_ph: train_label_batch})

                        log_eval_job(job_name, os.getpid(), assign_device)
                        acc_sum = 0
                        eval_batch_size = 50
                        num_batch_eval = eval_labels.shape[0] // eval_batch_size
                        for be in range(num_batch_eval):
                            # print('evaluation step %d / %d' % (i + 1, num_batch_eval))
                            batch_offset = be * eval_batch_size
                            batch_end = (be + 1) * eval_batch_size
                            eval_feature_batch = eval_feature[batch_offset:batch_end]
                            eval_label_batch = eval_labels[batch_offset:batch_end]
                            acc_batch = sess.run(eval_op, feed_dict={feature_ph: eval_feature_batch,
                                                                     label_ph: eval_label_batch})
                            acc_sum += acc_batch

                        cur_accuracy = acc_sum / num_batch_eval

                        pre_accuracy = self.job_dict_accuracy[job_name]

                        epoch_end_marker = perf_counter()

                        # tracking time and accuracy
                        self.job_dict_runtime[job_name] += epoch_end_marker - epoch_start_marker
                        self.job_dict_epoch[job_name] += 1
                        self.job_dict_accuracy[job_name] = cur_accuracy

                        if mode == SchedType.SCHED_RUNTIME:
                            job_progress = self.check_job_progress_runtime(job_name,
                                                                           cur_accuracy,
                                                                           job_slo_value,
                                                                           gpu_device)
                        else:
                            job_progress = self.check_job_progress_others(job_name,
                                                                          job_slo,
                                                                          job_slo_value,
                                                                          job_slo_max_time,
                                                                          pre_accuracy,
                                                                          cur_accuracy,
                                                                          gpu_device)

                        if job_progress == JobStatus.COMPLETE_ATTAIN:
                            saver.save(sess, checkpoint_file)
                            msg = 'job {} reaches SLO'.format(job_id)
                            return msg
                        elif job_progress == JobStatus.COMPLETE_UNATTAIN:
                            saver.save(sess, checkpoint_file)
                            msg = 'job {} is finished'.format(job_id)
                            return msg
                        else:
                            pass

                        if mode == SchedType.SCHED_OTHERS:
                            slot_end_marker = perf_counter()
                            running_slot_time = slot_end_marker - process_start_marker
        except RuntimeError:
            print('######### Unknown Error: {} #########'.format(job_name))
            self.gpu_slot_runtime[gpu_device] = 0
            self.sem_runtime.release()

    def process_job_runtime(self):
        self.sem_runtime.acquire()
        process_start_marker = perf_counter()

        gpu_device = -1
        while True:
            gpu_device += 1
            slot_idx = gpu_device % self.num_gpu
            if self.gpu_slot_runtime[slot_idx] == 0:
                gpu_device = slot_idx
                self.gpu_slot_runtime[slot_idx] = 1
                break

        self.ml_workload_runtime.sort(key=compared_item)
        try:
            job_data = self.ml_workload_runtime.pop()
        except IndexError:
            return

        job_model = job_data['model']

        if job_model == 'bert':
            self.train_bert_model(job_data,
                                  gpu_device,
                                  process_start_marker,
                                  mode=SchedType.SCHED_RUNTIME)

        elif job_model == 'lstm' or job_model == 'bilstm':
            self.train_lstm_model(job_data,
                                  gpu_device,
                                  process_start_marker,
                                  mode=SchedType.SCHED_RUNTIME)
        else:
            self.train_cv_model(job_data,
                                gpu_device,
                                process_start_marker,
                                mode=SchedType.SCHED_RUNTIME)

    def process_job_others(self):
        self.sem_others.acquire()
        process_start_marker = perf_counter()

        gpu_device = -1
        while True:
            gpu_device += 1
            slot_idx = gpu_device % self.num_gpu
            if self.gpu_slot_others[slot_idx] == 0:
                gpu_device = slot_idx
                self.gpu_slot_others[slot_idx] = 1
                break

        # get the job data from the queue
        try:
            job_data = self.job_queue_others.get_nowait()
        except queue.Empty:
            return

        job_id = job_data['id']
        job_model = job_data['model']

        if job_model == 'bert':
            self.train_bert_model(job_data,
                                  gpu_device,
                                  process_start_marker,
                                  mode=SchedType.SCHED_OTHERS)

        elif job_model == 'lstm' or job_model == 'bilstm':
            self.train_lstm_model(job_data,
                                  gpu_device,
                                  process_start_marker,
                                  mode=SchedType.SCHED_OTHERS)

        else:
            self.train_cv_model(job_data,
                                gpu_device,
                                process_start_marker,
                                mode=SchedType.SCHED_OTHERS)

        # exceed the running slot and haven't achieve goal so put the job back to the queue
        self.job_queue_others.put(job_data)

        msg = 'job {} is finished the current running slot'.format(job_id)
        self.gpu_slot_others[gpu_device] = 0
        self.sem_others.release()
        return msg

    def run(self):
        print('*********** start processing runtime slo jobs ***********')
        while len(self.ml_workload_runtime) != 0:
            results_runtime = list()
            runtime_job_num = len(self.ml_workload_runtime)
            for idx in range(runtime_job_num):
                result = self.proc_pool_runtime.apply_async(self.process_job_runtime)
                results_runtime.append(result)

            for i in results_runtime:
                i.wait()

            for i in results_runtime:
                if i.ready():
                    if i.successful():
                        print(i.get())

        print('*********** start processing other slo jobs ***********')

        while not self.job_queue_others.empty():
            results_others = list()
            # for idx in range(job_queue_others.qsize()):
            for idx in range(self.num_gpu):
                result = self.proc_pool_others.apply_async(self.process_job_others)
                results_others.append(result)

            for i in results_others:
                i.wait()

            for i in results_others:
                if i.ready():
                    if i.successful():
                        print(i.get())

    def output(self):
        #######################################################
        # printout the log information
        #######################################################

        for key in self.job_dict_accuracy:
            print('{} [accuracy]-> {}'.format(key, self.job_dict_accuracy[key]))

        for key in self.job_dict_runtime:
            print('{} [runtime]-> {}'.format(key, self.job_dict_runtime[key]))

        for key in self.job_dict_epoch:
            print('{} [epoch]-> {}'.format(key, self.job_dict_epoch[key]))

        for key in self.job_dict_completion_time:
            print('{} [completion time]-> {}'.format(key, self.job_dict_completion_time[key]))

        for key in self.job_dict_attainment:
            print('{} [attainment]-> {}'.format(key, self.job_dict_attainment[key]))

        print("show the history")

        for key in self.job_history_accuracy:
            print('{} [acc_history]-> {}'.format(key, self.job_history_accuracy[key]))

        for key in self.job_history_runtime:
            print('{} [runtime_history]-> {}'.format(key, self.job_history_runtime[key]))

        proc_end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print('============= the whole exp finish at: {}===================='.format(proc_end_time))
