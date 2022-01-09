import os
import queue
from datetime import datetime
import tensorflow as tf
import multiprocessing as mp
from time import perf_counter

from rotary.estimator.relaqs_estimator import ReLAQSEstimator
from rotary.estimator.rotary_estimator import RotaryEstimator
from rotary.common.constants import JobSLO, JobStatus, SchedType
from rotary.common.property_utils import PropertyUtils
from rotary.common.log_utils import log_get_job, log_train_job, log_eval_job, log_time_accuracy
from rotary.common.model_utils import build_cv_model, build_nlp_model
from rotary.common.sched_utils import (init_tf_config,
                                       init_tf_vars,
                                       get_bert_dataset,
                                       prepare_bert_dataset)

import rotary.reader.udtb_reader as udtb_reader
import rotary.reader.cifar_reader as cifar_reader


class Rotary:
    def __init__(self,
                 path_file,
                 para_file,
                 knowledgebase_path,
                 estimator,
                 ml_workload):
        # path config
        self.path_file = path_file
        self.para_file = para_file
        self.knowledgebase_path = knowledgebase_path

        if estimator == 'rotary':
            self.estimator = RotaryEstimator(topk=5, poly_deg=3)
        elif estimator == 'relaqs':
            self.estimator = ReLAQSEstimator()
        else:
            raise ValueError('estimator name is not recognized')

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
        self.sem_trial = mp.Semaphore(self.num_gpu)
        self.sem_rotary = mp.Semaphore(self.num_gpu)
        self.gpu_slot_trial = mp.Array('i', [0] * self.num_gpu)
        self.gpu_slot_rotary = mp.Array('i', [0] * self.num_gpu)

        self.start_time_overall = perf_counter()

        # queue for trial phrase
        self.job_queue_trial = mp.Manager().Queue()
        # queue for checking if all jobs are completed
        self.job_queue_anony = mp.Manager().Queue()

        # list for rotary phrase
        self.job_list_rotary = mp.Manager().list()
        # list for getting the total parameters of each job
        self.ml_workload_shared = mp.Manager().list()
        # the workload for processing
        self.ml_workload = ml_workload

        # dict for storing job's current accuracy
        self.job_dict_accuracy = mp.Manager().dict()
        # dict for storing job's overall running time
        self.job_dict_runtime = mp.Manager().dict()
        # dict for storing job's overall training epochs
        self.job_dict_epoch = mp.Manager().dict()
        # dict for storing parameters of all jobs
        self.job_dict_parameters = mp.Manager().dict()
        # dict for storing job's completion time (achieving SLO or being terminated)
        self.job_dict_completion_time = mp.Manager().dict()
        # dict for storing if the job achieving the SLO
        self.job_dict_attainment = mp.Manager().dict()
        # dict for storing the time of a single training epoch of each job
        self.job_dict_epochtime = mp.Manager().dict()

        # dict for storing job's progress
        self.job_dict_progress = dict()
        # dict for tracking epochs with wall-time
        self.job_history_runtime = dict()
        # dict for tracking accuracy with wall-time
        self.job_history_accuracy = dict()

        # init process pool
        self.proc_pool_trial = mp.Pool(self.num_gpu, maxtasksperchild=1)
        self.proc_pool_rotary = mp.Pool(self.num_gpu, maxtasksperchild=1)

        #######################################################
        # start processing the workload
        #######################################################

        proc_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print('============= the whole exp starts at: {}===================='.format(proc_start_time))

        # init some dicts to track the progress
        for job_item in ml_workload:
            job_item_key = str(job_item['id']) + '-' + job_item['model']
            self.job_queue_trial.put(job_item)
            self.job_list_rotary.append(job_item)
            self.job_queue_anony.put(job_item_key)
            self.ml_workload_shared.append(job_item)

            self.job_dict_progress[job_item_key] = 0
            self.job_dict_accuracy[job_item_key] = 0
            self.job_dict_runtime[job_item_key] = 0
            self.job_dict_epoch[job_item_key] = 0
            self.job_dict_epochtime[job_item_key] = 0
            self.job_dict_parameters[job_item_key] = 0
            self.job_dict_completion_time[job_item_key] = 0
            self.job_dict_attainment[job_item_key] = 0

            # init the dict for tracking
            init_sub_accuracy_list = mp.Manager().list()
            init_sub_accuracy_list.append('0:' + proc_start_time)
            self.job_history_accuracy[job_item_key] = init_sub_accuracy_list
            init_sub_runtime_list = mp.Manager().list()
            init_sub_runtime_list.append('0:' + proc_start_time)
            self.job_history_runtime[job_item_key] = init_sub_runtime_list

        # self.rotary_estimator = RotaryEstimator(topk=5)
        for f in os.listdir(knowledgebase_path):
            model_acc_file = knowledgebase_path + '/' + f
            self.estimator.import_knowledge_archive(model_acc_file)

    def complete_job(self, job_name, gpu_device, attain, mode):
        end_time_overall = perf_counter()
        self.job_dict_completion_time[job_name] = end_time_overall - self.start_time_overall
        self.job_dict_attainment[job_name] = attain
        if mode == SchedType.SCHED_TRIAL:
            self.gpu_slot_trial[gpu_device] = 0
            self.sem_trial.release()
        else:
            self.gpu_slot_rotary[gpu_device] = 0
            self.sem_rotary.release()

    def check_job_progress(self,
                           job_name,
                           job_slo,
                           job_slo_value,
                           job_slo_max_time,
                           pre_accuracy,
                           cur_accuracy,
                           gpu_device,
                           mode):
        log_time_accuracy(job_name,
                          cur_accuracy,
                          self.job_history_accuracy,
                          self.job_dict_epoch,
                          self.job_history_runtime)

        if job_slo == 'accuracy':
            if self.job_dict_accuracy[job_name] >= job_slo_value:
                self.complete_job(job_name, gpu_device, 1, mode)
                return JobStatus.COMPLETE_ATTAIN
            elif self.job_dict_epoch[job_name] >= job_slo_max_time:
                self.complete_job(job_name, gpu_device, 0, mode)
                return JobStatus.COMPLETE_UNATTAIN
            else:
                return JobStatus.INCOMPLETE
        elif job_slo == 'convergence':
            delta = round(abs(cur_accuracy - pre_accuracy), 4)
            if delta <= job_slo_value:
                self.complete_job(job_name, gpu_device, 1, mode)
                return JobStatus.COMPLETE_ATTAIN
            elif self.job_dict_epoch[job_name] >= job_slo_max_time:
                self.complete_job(job_name, gpu_device, 0, mode)
                return JobStatus.COMPLETE_ATTAIN
            else:
                return JobStatus.INCOMPLETE
        elif job_slo == 'runtime':
            if self.job_dict_epoch[job_name] >= job_slo_value:
                self.complete_job(job_name, gpu_device, 1, mode)
                return JobStatus.COMPLETE_ATTAIN
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

        # Params for bert model and tokenization
        bert_path = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
        max_seq_length = 128
        offset = 500

        train_text, train_label = get_bert_dataset(max_seq_length)

        # start processing on the assigned device
        with tf.device(assign_device):
            with tf.Session(config=init_tf_config()) as sess:
                (
                    train_input_ids,
                    train_input_masks,
                    train_segment_ids,
                    train_labels
                ) = prepare_bert_dataset(bert_path, sess, train_text, train_label, max_seq_length)

                # build the model
                model = build_nlp_model(model_type=job_data['model'],
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
                self.job_dict_runtime[job_name] += preparation_end_marker - process_start_marker

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

                    # add current results
                    if isinstance(self.estimator, RotaryEstimator):
                        self.estimator.import_knowledge_realtime(job_name,
                                                                 cur_accuracy,
                                                                 self.job_dict_epoch[job_name])

                    job_progress = self.check_job_progress(job_name,
                                                           job_slo,
                                                           job_slo_value,
                                                           job_slo_max_time,
                                                           pre_accuracy,
                                                           cur_accuracy,
                                                           gpu_device,
                                                           mode)

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

                    if mode == SchedType.SCHED_ROTARY:
                        slot_end_marker = perf_counter()
                        running_slot_time = slot_end_marker - process_start_marker
                    else:
                        running_slot_time = self.running_slot

                logit.save(model_ckpt_save_path + '/' + job_name + '.h5')

    def train_lstm_model(self,
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
            logit, _ = model.build(word2index, tag2index)

            # load the checkpoint if it exists
            if os.path.exists(model_ckpt_save_path + '/' + job_name + '.h5'):
                logit.load_weights(model_ckpt_save_path + '/' + job_name + '.h5')

            with tf.Session(config=init_tf_config()) as sess:
                sess.run(tf.global_variables_initializer())

                # add the prepare time for this process
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

                    # add current results
                    if isinstance(self.estimator, RotaryEstimator):
                        self.estimator.import_knowledge_realtime(job_name,
                                                                 cur_accuracy,
                                                                 self.job_dict_epoch[job_name])

                    job_progress = self.check_job_progress(job_name,
                                                           job_slo,
                                                           job_slo_value,
                                                           job_slo_max_time,
                                                           pre_accuracy,
                                                           cur_accuracy,
                                                           gpu_device,
                                                           mode)

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

                    if mode == SchedType.SCHED_ROTARY:
                        slot_end_marker = perf_counter()
                        running_slot_time = slot_end_marker - process_start_marker
                    else:
                        running_slot_time = self.running_slot

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

        with tf.device(assign_device):
            feature_ph = tf.placeholder(tf.float32, [None, 32, 32, 3])
            label_ph = tf.placeholder(tf.int64, [None, 10])
            train_op, eval_op, _ = build_cv_model(job_data,
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

                preparation_end_marker = perf_counter()
                # add the prepare time for this process
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

                    # tracking the time and accuracy
                    self.job_dict_runtime[job_name] += epoch_end_marker - epoch_start_marker
                    self.job_dict_epoch[job_name] += 1
                    self.job_dict_accuracy[job_name] = cur_accuracy

                    # add current results
                    if isinstance(self.estimator, RotaryEstimator):
                        self.estimator.import_knowledge_realtime(job_name,
                                                                 cur_accuracy,
                                                                 self.job_dict_epoch[job_name])

                    job_progress = self.check_job_progress(job_name,
                                                           job_slo,
                                                           job_slo_value,
                                                           job_slo_max_time,
                                                           pre_accuracy,
                                                           cur_accuracy,
                                                           gpu_device,
                                                           mode)

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

                    if mode == SchedType.SCHED_ROTARY:
                        slot_end_marker = perf_counter()
                        running_slot_time = slot_end_marker - process_start_marker
                    else:
                        running_slot_time = self.running_slot

                saver.save(sess, checkpoint_file)

    def process_job_trial(self):
        self.sem_trial.acquire()
        process_start_marker = perf_counter()

        gpu_device = -1
        while True:
            gpu_device += 1
            slot_idx = gpu_device % self.num_gpu
            if self.gpu_slot_trial[slot_idx] == 0:
                gpu_device = slot_idx
                self.gpu_slot_trial[slot_idx] = 1
                break

        try:
            job_data = self.job_queue_trial.get_nowait()
        except queue.Empty:
            return

        job_id = job_data['id']
        job_model = job_data['model']

        if job_model == 'bert':
            self.train_bert_model(job_data,
                                  gpu_device,
                                  process_start_marker,
                                  mode=SchedType.SCHED_TRIAL)

        elif job_model == 'lstm' or job_model == 'bilstm':
            self.train_lstm_model(job_data,
                                  gpu_device,
                                  process_start_marker,
                                  mode=SchedType.SCHED_TRIAL)

        else:
            self.train_cv_model(job_data,
                                gpu_device,
                                process_start_marker,
                                mode=SchedType.SCHED_TRIAL)

        msg_trial = 'job {} is finished the current running slot'.format(job_id)
        self.gpu_slot_trial[gpu_device] = 0
        self.sem_trial.release()
        return msg_trial

    def process_job(self, job_data):
        self.sem_rotary.acquire()
        process_start_marker = perf_counter()

        gpu_device = -1
        while True:
            gpu_device += 1
            slot_idx = gpu_device % self.num_gpu
            if self.gpu_slot_rotary[slot_idx] == 0:
                gpu_device = slot_idx
                self.gpu_slot_rotary[slot_idx] = 1
                break

        # randomly pop a job key from the queue_anony
        try:
            job_anony = self.job_queue_anony.get_nowait()
        except queue.Empty:
            return

        job_id = job_data['id']
        job_model = job_data['model']

        # handling different jobs
        if job_model == 'bert':
            self.train_bert_model(job_data,
                                  gpu_device,
                                  process_start_marker,
                                  mode=SchedType.SCHED_ROTARY)

        elif job_model == 'lstm' or job_model == 'bilstm':
            self.train_lstm_model(job_data,
                                  gpu_device,
                                  process_start_marker,
                                  mode=SchedType.SCHED_ROTARY)

        else:
            self.train_cv_model(job_data,
                                gpu_device,
                                process_start_marker,
                                mode=SchedType.SCHED_ROTARY)

        # exceed the running slot and haven't achieve goal so put the job back to the queue
        self.job_list_rotary.append(job_data)
        self.job_queue_anony.put(job_anony)
        msg_slot = 'job {} is finished the current running slot'.format(job_id)

        self.gpu_slot_rotary[gpu_device] = 0
        self.sem_rotary.release()
        return msg_slot

    def run(self):
        #######################################################
        # start the trial process
        #######################################################

        results_trial = list()
        for idx in range(len(self.ml_workload)):
            result = self.proc_pool_trial.apply_async(self.process_job_trial)
            results_trial.append(result)

        for i in results_trial:
            i.wait()

        for i in results_trial:
            if i.ready():
                if i.successful():
                    print(i.get())

        for key in self.job_dict_accuracy:
            print(key, '[accuracy]->', self.job_dict_accuracy[key])

        for key in self.job_dict_runtime:
            print(key, '[time]->', self.job_dict_runtime[key])

        for key in self.job_dict_epoch:
            print(key, '[epoch]->', self.job_dict_epoch[key])

        for key in self.job_dict_epochtime:
            print(key, '[epoch_time]->', self.job_dict_epochtime[key])

        for ml_job in self.ml_workload_shared:
            print('Job {}: {} parameters'.format(ml_job['id'], ml_job['num_parameters']))

        #######################################################
        # start the rotary process
        #######################################################

        fairness = True
        while not self.job_queue_anony.empty():
            results_rotary = list()
            job_select = self.job_list_rotary[0]
            print('************* current rotary queue: {} *************'.format(self.job_queue_anony.qsize()))
            for gpu_idx in range(self.num_gpu):
                r_score_mark = float('inf') if fairness else float('-inf')

                for job_ins in self.job_list_rotary:
                    job_ins_key = str(job_ins['id']) + '-' + job_ins['model']

                    job_ins_slo = job_ins['goal_type']
                    job_ins_slo_value = job_ins['goal_value']

                    if job_ins_slo == 'runtime':
                        current_epoch = self.job_dict_epoch[job_ins_key]
                        r_score = current_epoch / job_ins_slo_value
                        self.job_dict_progress[job_ins_key] = r_score

                    elif job_ins_slo == 'accuracy':
                        job_ins_slo_max_time = job_ins['goal_value_extra']
                        current_epoch = self.job_dict_epoch[job_ins_key]
                        estimate_all_epoch = round(self.estimator.predict(job_ins, job_ins_slo_value, 'epoch'))
                        if estimate_all_epoch > job_ins_slo_max_time:
                            r_score = current_epoch / job_ins_slo_max_time
                        else:
                            r_score = current_epoch / estimate_all_epoch if estimate_all_epoch != 0 else 0
                        self.job_dict_progress[job_ins_key] = r_score

                    elif job_ins_slo == 'convergence':
                        job_ins_slo_max_time = job_ins['goal_value_extra']
                        current_epoch = self.job_dict_epoch[job_ins_key]
                        current_accuracy = self.job_dict_accuracy[job_ins_key]
                        expected_accuracy = current_accuracy + job_ins_slo_value
                        estimate_all_epoch = round(self.estimator.predict(job_ins, expected_accuracy, 'epoch'))
                        if estimate_all_epoch <= 0:
                            r_score = 1
                        else:
                            if estimate_all_epoch > job_ins_slo_max_time:
                                r_score = current_epoch / job_ins_slo_max_time
                            else:
                                r_score = current_epoch / estimate_all_epoch

                        self.job_dict_progress[job_ins_key] = r_score

                    else:
                        raise ValueError('the job objective type is not supported')

                    if fairness:
                        if r_score_mark > r_score:
                            r_score_mark = r_score
                            job_select = job_ins
                    else:
                        if r_score_mark <= r_score:
                            r_score_mark = r_score
                            job_select = job_ins

                print('$$$$$ JOB SELECTION: {} $$$$$'.format(job_select))

                try:
                    self.job_list_rotary.remove(job_select)
                except ValueError:
                    msg_err = 'job has been handled by other GPU'
                    print(msg_err)
                    continue

                result = self.proc_pool_rotary.apply_async(self.process_job, args=job_select)
                results_rotary.append(result)

            for i in results_rotary:
                i.wait()

            for i in results_rotary:
                if i.ready():
                    if i.successful():
                        print(i.get())

            fairness = False
            threshold = 0.5
            for job in self.job_list_rotary:
                job_key = str(job['id']) + '-' + job['model']
                if self.job_dict_progress[job_key] < threshold:
                    fairness = True
                    break

            if fairness:
                print('||||||||||||||||||| USING FAIRNESS POLICY |||||||||||||||||||')
            else:
                print('||||||||||||||||||| USING AGGRESSIVE POLICY |||||||||||||||||||')

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

        # print('total log count: {}'.format(log_func.log_counter.value))
        # print('log time: {}'.format(log_func.log_time.value))
        # print('rotary time: {}'.format(log_func.rotary_time.value))
