import tensorflow as tf
import multiprocessing as mp
import numpy as np
from timeit import default_timer as timer
import os
import queue
from datetime import datetime

import config.config_rotary as cfg_rotary
import config.config_path as cfg_path
import workload.tensorflow_cifar.tools.cifar_reader as cifar_reader
import workload.tensorflow_nlp.tools.udtb_reader as udtb_reader
import workload.tensorflow_nlp.tools.lmrd_reader as lmrd_reader
from estimator.dl_estimator import DLEstimator
from workload.workload_generator import WorkloadGenerator
import utils.log_func as log_func
from utils.model_tool import build_cv_model, build_nlp_model
from utils.tf_func import initialize_config, initialize_vars

sem_trial = mp.Semaphore(cfg_rotary.num_gpu)
sem_rotary = mp.Semaphore(cfg_rotary.num_gpu)
gpu_slot_trial = mp.Array('i', [0] * cfg_rotary.num_gpu)
gpu_slot_rotary = mp.Array('i', [0] * cfg_rotary.num_gpu)


def train_job_trial(shared_runtime_history,
                    shared_accuracy_history):
    sem_trial.acquire()
    trial_slot_start_marker = timer()

    gpu_device = -1
    while True:
        gpu_device += 1
        slot_idx = gpu_device % num_gpu
        if gpu_slot_trial[slot_idx] == 0:
            gpu_device = slot_idx
            gpu_slot_trial[slot_idx] = 1
            break

    try:
        job_data = job_queue_trial.get_nowait()
    except queue.Empty:
        return

    job_id = job_data['id']
    job_model = job_data['model']
    job_name = str(job_id) + '-' + job_model

    assign_device = '/gpu:' + str(gpu_device)
    log_func.log_get_job(job_name, os.getpid(), assign_device)

    model_opt = job_data['opt']
    model_learn_rate = job_data['learn_rate']
    train_batchsize = job_data['batch_size']

    job_slo = job_data['goal_type']
    job_slo_value = job_data['goal_value']
    job_slo_max_time = -1
    if job_slo == 'accuracy' or job_slo == 'convergence':
        job_slo_max_time = job_data['goal_value_extra']

    if job_model == 'bert':
        # Params for bert model and tokenization
        bert_path = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
        max_seq_length = 128
        offset = 500

        train_df = lmrd_reader.download_and_load_datasets()

        # Create datasets (Only take up to max_seq_length words for memory)
        train_text = train_df["sentence"].tolist()
        train_text = [" ".join(t.split()[0:max_seq_length]) for t in train_text]
        train_text = np.array(train_text, dtype=object)[:, np.newaxis]
        train_label = train_df["polarity"].tolist()

        # start processing on the assigned device
        with tf.device(assign_device):
            # create the folder for saving models
            model_ckpt_save_path = cfg_path.ckpt_save_path + '/' + job_name
            if not os.path.exists(model_ckpt_save_path):
                os.makedirs(model_ckpt_save_path)

            with tf.Session(config=initialize_config()) as sess:
                # Instantiate tokenizer
                tokenizer = lmrd_reader.create_tokenizer_from_hub_module(bert_path, sess)
                # Convert data to InputExample format
                train_examples = lmrd_reader.convert_text_to_examples(train_text, train_label)
                # Convert to features
                (
                    train_input_ids,
                    train_input_masks,
                    train_segment_ids,
                    train_labels,
                ) = lmrd_reader.convert_examples_to_features(tokenizer, train_examples, max_seq_length=max_seq_length)

                # build the model
                model = build_nlp_model(model_type=job_model,
                                        max_length=128,
                                        opt=model_opt,
                                        lr=model_learn_rate)
                logit, total_parameters = model.build()

                # store the total parameters of the model to dict
                job_parameters_dict[job_name] = total_parameters
                current_job = ml_workload_shared[int(job_id)]
                current_job['num_parameters'] = total_parameters
                ml_workload_shared[int(job_id)] = current_job

                # Instantiate variables
                initialize_vars(sess)

                epochtime_start_marker = timer()

                log_func.log_start_train(job_name, os.getpid(), assign_device)
                logit.fit([train_input_ids, train_input_masks, train_segment_ids],
                          train_labels,
                          epochs=1,
                          batch_size=train_batchsize,
                          verbose=0)

                # start evaluation phrase
                log_func.log_start_eval(job_name, os.getpid(), assign_device)
                scores = logit.evaluate([train_input_ids[0:offset],
                                         train_input_masks[0:offset],
                                         train_segment_ids[0:offset]],
                                        train_labels[0:offset],
                                        verbose=0)
                cur_accuracy = scores[1]
                log_func.log_end_eval(job_name, cur_accuracy, assign_device)

                pre_accuracy = job_accuracy_dict[job_name]

                # compute the time of single epoch time for the job
                epochtime_end_marker = timer()
                job_epochtime_dict[job_name] = epochtime_end_marker - epochtime_start_marker

                # record the time of training slot
                trial_slot_end_marker = timer()
                job_runtime_dict[job_name] += trial_slot_end_marker - trial_slot_start_marker

                # record the meta information
                job_epoch_dict[job_name] += 1
                job_accuracy_dict[job_name] = cur_accuracy

                # add this trial result to estimator
                dl_estimator.add_actual_data(job_key=job_name, accuracy=cur_accuracy, epoch=1)

                if job_slo == 'accuracy':
                    if job_accuracy_dict[job_name] >= job_slo_value:
                        end_time_overall = timer()
                        job_completion_time_dict[job_name] = end_time_overall - start_time_overall
                        job_attain_dict[job_name] = 1
                        log_func.log_time_accuracy(job_name,
                                                   cur_accuracy,
                                                   shared_runtime_history,
                                                   job_epoch_dict,
                                                   shared_accuracy_history)
                        logit.save(model_ckpt_save_path + '/' + job_name + '.h5')
                        msg_trial = 'job {} reaches SLO'.format(job_id)
                        gpu_slot_trial[gpu_device] = 0
                        sem_trial.release()
                        return msg_trial

                    if job_epoch_dict[job_name] >= job_slo_max_time:
                        end_time_overall = timer()
                        job_completion_time_dict[job_name] = end_time_overall - start_time_overall
                        log_func.log_time_accuracy(job_name,
                                                   cur_accuracy,
                                                   shared_runtime_history,
                                                   job_epoch_dict,
                                                   shared_accuracy_history)
                        logit.save(model_ckpt_save_path + '/' + job_name + '.h5')
                        msg_trial = 'job {} is finished'.format(job_id)
                        gpu_slot_trial[gpu_device] = 0
                        sem_trial.release()
                        return msg_trial

                elif job_slo == 'convergence':
                    delta = round(abs(cur_accuracy - pre_accuracy), 4)
                    if delta <= job_slo_value:
                        end_time_overall = timer()
                        job_completion_time_dict[job_name] = end_time_overall - start_time_overall
                        job_attain_dict[job_name] = 1
                        log_func.log_time_accuracy(job_name,
                                                   cur_accuracy,
                                                   shared_runtime_history,
                                                   job_epoch_dict,
                                                   shared_accuracy_history)
                        logit.save(model_ckpt_save_path + '/' + job_name + '.h5')
                        msg_trial = 'job {} reaches the SLO'.format(job_id)
                        gpu_slot_trial[gpu_device] = 0
                        sem_trial.release()
                        return msg_trial

                    if job_epoch_dict[job_name] >= job_slo_max_time:
                        end_time_overall = timer()
                        job_completion_time_dict[job_name] = end_time_overall - start_time_overall
                        log_func.log_time_accuracy(job_name,
                                                   cur_accuracy,
                                                   shared_runtime_history,
                                                   job_epoch_dict,
                                                   shared_accuracy_history)
                        logit.save(model_ckpt_save_path + '/' + job_name + '.h5')
                        msg_trial = 'job {} is finished'.format(job_id)
                        gpu_slot_trial[gpu_device] = 0
                        sem_trial.release()
                        return msg_trial

                elif job_slo == 'runtime':
                    if job_epoch_dict[job_name] >= job_slo_value:
                        end_time_overall = timer()
                        job_completion_time_dict[job_name] = end_time_overall - start_time_overall
                        job_attain_dict[job_name] = 1
                        log_func.log_time_accuracy(job_name,
                                                   cur_accuracy,
                                                   shared_runtime_history,
                                                   job_epoch_dict,
                                                   shared_accuracy_history)
                        logit.save(model_ckpt_save_path + '/' + job_name + '.h5')
                        msg_trial = 'job {} reaches the SLO'.format(job_id)
                        gpu_slot_trial[gpu_device] = 0
                        sem_trial.release()
                        return msg_trial
                else:
                    raise ValueError('the job objective type is not supported')

                # save the model and exit the trial slot
                logit.save(model_ckpt_save_path + '/' + job_name + '.h5')
                log_func.log_time_accuracy(job_name,
                                           cur_accuracy,
                                           shared_runtime_history,
                                           job_epoch_dict,
                                           shared_accuracy_history)

    elif job_model == 'lstm' or job_model == 'bilstm':
        (train_sentences_x,
         val_sentences_x,
         train_tags_y,
         val_tags_y,
         MAX_LENGTH,
         word2index,
         tag2index) = udtb_reader.load_udtb_dataset()

        # start processing on the assigned device
        with tf.device(assign_device):
            # create the folder for saving models
            model_ckpt_save_path = cfg_path.ckpt_save_path + '/' + job_name
            if not os.path.exists(model_ckpt_save_path):
                os.makedirs(model_ckpt_save_path)

            # build model
            model = build_nlp_model(model_type=job_model,
                                    max_length=MAX_LENGTH,
                                    opt=model_opt,
                                    lr=model_learn_rate)
            logit, total_parameters = model.build(word2index, tag2index)

            # store the total parameters of the model to dict
            job_parameters_dict[job_name] = total_parameters
            current_job = ml_workload_shared[int(job_id)]
            current_job['num_parameters'] = total_parameters
            ml_workload_shared[int(job_id)] = current_job

            with tf.Session(config=initialize_config()) as sess:
                sess.run(tf.global_variables_initializer())

                epochtime_start_marker = timer()
                log_func.log_start_train(job_name, os.getpid(), assign_device)
                logit.fit(train_sentences_x,
                          udtb_reader.to_categorical(train_tags_y, len(tag2index)),
                          batch_size=train_batchsize,
                          epochs=1,
                          verbose=0)

                # start evaluation phrase
                log_func.log_start_eval(job_name, os.getpid(), assign_device)
                scores = logit.evaluate(val_sentences_x,
                                        udtb_reader.to_categorical(val_tags_y, len(tag2index)),
                                        verbose=0)
                cur_accuracy = scores[1]
                log_func.log_end_eval(job_name, cur_accuracy, assign_device)

                pre_accuracy = job_accuracy_dict[job_name]

                # record the time of single epoch time for the job
                epochtime_end_marker = timer()
                job_epochtime_dict[job_name] = epochtime_end_marker - epochtime_start_marker

                # record the time of the training slot
                trial_slot_end_marker = timer()
                job_runtime_dict[job_name] += trial_slot_end_marker - trial_slot_start_marker

                # record some meta information
                job_epoch_dict[job_name] += 1
                job_accuracy_dict[job_name] = cur_accuracy

                # add this trial result to estimator
                dl_estimator.add_actual_data(job_key=job_name, accuracy=cur_accuracy, epoch=1)

                if job_slo == 'accuracy':
                    if job_accuracy_dict[job_name] >= job_slo_value:
                        end_time_overall = timer()
                        job_completion_time_dict[job_name] = end_time_overall - start_time_overall
                        job_attain_dict[job_name] = 1
                        log_func.log_time_accuracy(job_name,
                                                   cur_accuracy,
                                                   shared_runtime_history,
                                                   job_epoch_dict,
                                                   shared_accuracy_history)
                        logit.save(model_ckpt_save_path + '/' + job_name + '.h5')
                        msg_trial = 'job {} reaches SLO'.format(job_id)
                        gpu_slot_trial[gpu_device] = 0
                        sem_trial.release()
                        return msg_trial

                    if job_epoch_dict[job_name] >= job_slo_max_time:
                        end_time_overall = timer()
                        job_completion_time_dict[job_name] = end_time_overall - start_time_overall
                        log_func.log_time_accuracy(job_name,
                                                   cur_accuracy,
                                                   shared_runtime_history,
                                                   job_epoch_dict,
                                                   shared_accuracy_history)
                        logit.save(model_ckpt_save_path + '/' + job_name + '.h5')
                        msg_trial = 'job {} is finished'.format(job_id)
                        gpu_slot_trial[gpu_device] = 0
                        sem_trial.release()
                        return msg_trial

                elif job_slo == 'convergence':
                    delta = round(abs(cur_accuracy - pre_accuracy), 4)
                    if delta <= job_slo_value:
                        end_time_overall = timer()
                        job_completion_time_dict[job_name] = end_time_overall - start_time_overall
                        job_attain_dict[job_name] = 1
                        log_func.log_time_accuracy(job_name,
                                                   cur_accuracy,
                                                   shared_runtime_history,
                                                   job_epoch_dict,
                                                   shared_accuracy_history)
                        logit.save(model_ckpt_save_path + '/' + job_name + '.h5')
                        msg_trial = 'job {} reaches the SLO'.format(job_id)
                        gpu_slot_trial[gpu_device] = 0
                        sem_trial.release()
                        return msg_trial

                    if job_epoch_dict[job_name] >= job_slo_max_time:
                        end_time_overall = timer()
                        job_completion_time_dict[job_name] = end_time_overall - start_time_overall
                        log_func.log_time_accuracy(job_name,
                                                   cur_accuracy,
                                                   shared_runtime_history,
                                                   job_epoch_dict,
                                                   shared_accuracy_history)
                        logit.save(model_ckpt_save_path + '/' + job_name + '.h5')
                        msg_trial = 'job {} is finished'.format(job_id)
                        gpu_slot_trial[gpu_device] = 0
                        sem_trial.release()
                        return msg_trial

                elif job_slo == 'runtime':
                    if job_epoch_dict[job_name] >= job_slo_value:
                        end_time_overall = timer()
                        job_completion_time_dict[job_name] = end_time_overall - start_time_overall
                        job_attain_dict[job_name] = 1
                        log_func.log_time_accuracy(job_name,
                                                   cur_accuracy,
                                                   shared_runtime_history,
                                                   job_epoch_dict,
                                                   shared_accuracy_history)
                        logit.save(model_ckpt_save_path + '/' + job_name + '.h5')
                        msg_trial = 'job {} reaches the SLO'.format(job_id)
                        gpu_slot_trial[gpu_device] = 0
                        sem_trial.release()
                        return msg_trial
                else:
                    raise ValueError('the job objective type is not supported')

                # save the model and exit the trial slot
                logit.save(model_ckpt_save_path + '/' + job_name + '.h5')
                log_func.log_time_accuracy(job_name,
                                           cur_accuracy,
                                           shared_runtime_history,
                                           job_epoch_dict,
                                           shared_accuracy_history)

    else:
        img_w = 32
        img_h = 32
        num_chn = 3
        num_cls = 10

        train_feature, train_labels, eval_feature, eval_labels = cifar_reader.load_cifar10_keras()

        with tf.device(assign_device):
            feature_ph = tf.placeholder(tf.float32, [None, img_w, img_h, num_chn])
            label_ph = tf.placeholder(tf.int64, [None, num_cls])
            train_op, eval_op, total_parameters = build_cv_model(job_data,
                                                                 model_opt,
                                                                 model_learn_rate,
                                                                 num_cls,
                                                                 feature_ph,
                                                                 label_ph)

            # store the total parameters of the model to dict
            job_parameters_dict[job_name] = total_parameters
            current_job = ml_workload_shared[int(job_id)]
            current_job['num_parameters'] = total_parameters
            ml_workload_shared[int(job_id)] = current_job

            # ready to train the job
            saver = tf.train.Saver()
            model_ckpt_save_path = cfg_path.ckpt_save_path + '/' + job_name
            if not os.path.exists(model_ckpt_save_path):
                os.makedirs(model_ckpt_save_path)
            checkpoint_file = model_ckpt_save_path + '/model_ckpt'

            with tf.Session(config=initialize_config()) as sess:
                sess.run(tf.global_variables_initializer())

                num_batch = train_labels.shape[0] // train_batchsize

                epochtime_start_marker = timer()
                log_func.log_start_train(job_name, os.getpid(), assign_device)
                for n in range(num_batch):
                    batch_offset = n * train_batchsize
                    batch_end = (n + 1) * train_batchsize

                    train_data_batch = train_feature[batch_offset:batch_end]
                    train_label_batch = train_labels[batch_offset:batch_end]

                    sess.run(train_op, feed_dict={feature_ph: train_data_batch, label_ph: train_label_batch})

                # start evaluation phrase
                log_func.log_start_eval(job_name, os.getpid(), assign_device)
                acc_sum = 0
                eval_batch_size = 50
                num_batch_eval = eval_labels.shape[0] // eval_batch_size
                for ne in range(num_batch_eval):
                    batch_offset = ne * eval_batch_size
                    batch_end = (ne + 1) * eval_batch_size
                    eval_feature_batch = eval_feature[batch_offset:batch_end]
                    eval_label_batch = eval_labels[batch_offset:batch_end]
                    acc_batch = sess.run(eval_op,
                                         feed_dict={feature_ph: eval_feature_batch, label_ph: eval_label_batch})
                    acc_sum += acc_batch

                cur_accuracy = acc_sum / num_batch_eval
                log_func.log_end_eval(job_name, cur_accuracy, assign_device)

                pre_accuracy = job_accuracy_dict[job_name]

                # record the epoch time
                epochtime_end_marker = timer()
                job_epochtime_dict[job_name] = epochtime_end_marker - epochtime_start_marker

                # record the some meta information
                trial_slot_end_marker = timer()
                job_runtime_dict[job_name] += trial_slot_end_marker - trial_slot_start_marker
                job_epoch_dict[job_name] += 1
                job_accuracy_dict[job_name] = cur_accuracy

                # add this trial result to estimator
                dl_estimator.add_actual_data(job_key=job_name, accuracy=cur_accuracy, epoch=1)

                if job_slo == 'accuracy':
                    if job_accuracy_dict[job_name] >= job_slo_value:
                        end_time_overall = timer()
                        job_completion_time_dict[job_name] = end_time_overall - start_time_overall
                        job_attain_dict[job_name] = 1
                        log_func.log_time_accuracy(job_name,
                                                   cur_accuracy,
                                                   shared_runtime_history,
                                                   job_epoch_dict,
                                                   shared_accuracy_history)
                        saver.save(sess, checkpoint_file)
                        msg_trial = 'job {} reaches SLO'.format(job_id)
                        gpu_slot_trial[gpu_device] = 0
                        sem_trial.release()
                        return msg_trial

                    if job_epoch_dict[job_name] >= job_slo_max_time:
                        end_time_overall = timer()
                        job_completion_time_dict[job_name] = end_time_overall - start_time_overall
                        log_func.log_time_accuracy(job_name,
                                                   cur_accuracy,
                                                   shared_runtime_history,
                                                   job_epoch_dict,
                                                   shared_accuracy_history)
                        saver.save(sess, checkpoint_file)
                        msg_trial = 'job {} is finished'.format(job_id)
                        gpu_slot_trial[gpu_device] = 0
                        sem_trial.release()
                        return msg_trial

                elif job_slo == 'convergence':
                    delta = round(abs(cur_accuracy - pre_accuracy), 4)
                    if delta <= job_slo_value:
                        end_time_overall = timer()
                        job_completion_time_dict[job_name] = end_time_overall - start_time_overall
                        job_attain_dict[job_name] = 1
                        log_func.log_time_accuracy(job_name,
                                                   cur_accuracy,
                                                   shared_runtime_history,
                                                   job_epoch_dict,
                                                   shared_accuracy_history)
                        saver.save(sess, checkpoint_file)
                        msg_trial = 'job {} reaches the SLO'.format(job_id)
                        gpu_slot_trial[gpu_device] = 0
                        sem_trial.release()
                        return msg_trial

                    if job_epoch_dict[job_name] >= job_slo_max_time:
                        end_time_overall = timer()
                        job_completion_time_dict[job_name] = end_time_overall - start_time_overall
                        log_func.log_time_accuracy(job_name,
                                                   cur_accuracy,
                                                   shared_runtime_history,
                                                   job_epoch_dict,
                                                   shared_accuracy_history)
                        saver.save(sess, checkpoint_file)
                        msg_trial = 'job {} is finished'.format(job_id)
                        gpu_slot_trial[gpu_device] = 0
                        sem_trial.release()
                        return msg_trial

                elif job_slo == 'runtime':
                    if job_epoch_dict[job_name] >= job_slo_value:
                        end_time_overall = timer()
                        job_completion_time_dict[job_name] = end_time_overall - start_time_overall
                        job_attain_dict[job_name] = 1
                        log_func.log_time_accuracy(job_name,
                                                   cur_accuracy,
                                                   shared_runtime_history,
                                                   job_epoch_dict,
                                                   shared_accuracy_history)
                        saver.save(sess, checkpoint_file)
                        msg_trial = 'job {} reaches the SLO'.format(job_id)
                        gpu_slot_trial[gpu_device] = 0
                        sem_trial.release()
                        return msg_trial

                else:
                    raise ValueError('the job objective type is not supported')

                # save the model and exit the trial slot
                saver.save(sess, checkpoint_file)
                log_func.log_time_accuracy(job_name,
                                           cur_accuracy,
                                           shared_runtime_history,
                                           job_epoch_dict,
                                           shared_accuracy_history)

    msg_trial = 'job {} is finished the current running slot'.format(job_id)
    gpu_slot_trial[gpu_device] = 0
    sem_trial.release()
    return msg_trial


def train_job(job_data,
              shared_runtime_history,
              shared_accuracy_history):
    sem_rotary.acquire()
    preparation_start_marker = timer()
    slot_start_marker = timer()
    # gpu_device = slot_idx

    gpu_device = -1
    while True:
        gpu_device += 1
        slot_idx = gpu_device % num_gpu
        if gpu_slot_rotary[slot_idx] == 0:
            gpu_device = slot_idx
            gpu_slot_rotary[slot_idx] = 1
            break

    # count the training time of this slot
    running_slot_time = 0

    # randomly pop a job key from the queue_anony
    try:
        job_anony = job_queue_anony.get_nowait()
    except queue.Empty:
        return

    job_id = job_data['id']
    job_model = job_data['model']
    job_name = str(job_id) + '-' + job_model
    assign_device = '/gpu:' + str(gpu_device)
    log_func.log_get_job(job_name, os.getpid(), assign_device)

    model_opt = job_data['opt']
    model_learn_rate = job_data['learn_rate']
    train_batchsize = job_data['batch_size']

    job_slo = job_data['goal_type']
    job_slo_value = job_data['goal_value']
    job_slo_max_time = -1
    if job_slo == 'accuracy' or job_slo == 'convergence':
        job_slo_max_time = job_data['goal_value_extra']

    # handling different jobs
    if job_model == 'bert':
        # Params for bert model and tokenization
        bert_path = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
        max_seq_length = 128
        offset = 500

        train_df = lmrd_reader.download_and_load_datasets()

        # Create datasets (Only take up to max_seq_length words for memory)
        train_text = train_df["sentence"].tolist()
        train_text = [" ".join(t.split()[0:max_seq_length]) for t in train_text]
        train_text = np.array(train_text, dtype=object)[:, np.newaxis]
        train_label = train_df["polarity"].tolist()

        # start processing on the assigned device
        with tf.device(assign_device):
            with tf.Session(config=initialize_config()) as sess:
                # Instantiate tokenizer
                tokenizer = lmrd_reader.create_tokenizer_from_hub_module(bert_path, sess)
                # Convert data to InputExample format
                train_examples = lmrd_reader.convert_text_to_examples(train_text, train_label)
                # Convert to features
                (
                    train_input_ids,
                    train_input_masks,
                    train_segment_ids,
                    train_labels
                ) = lmrd_reader.convert_examples_to_features(tokenizer, train_examples, max_seq_length=max_seq_length)

                # build the model
                model = build_nlp_model(model_type=job_model,
                                        max_length=128,
                                        opt=model_opt,
                                        lr=model_learn_rate)
                logit, _ = model.build()

                # load the checkpoint if it exists
                model_ckpt_save_path = cfg_path.ckpt_save_path + '/' + job_name
                if os.path.exists(model_ckpt_save_path + '/' + job_name + '.h5'):
                    logit.load_weights(model_ckpt_save_path + '/' + job_name + '.h5')

                # Instantiate variables
                initialize_vars(sess)

                # add the prepare time for this process
                preparation_end_marker = timer()
                job_runtime_dict[job_name] += preparation_end_marker - preparation_start_marker

                # check if the total runtime is less than running_slot
                while running_slot_time < running_slot:
                    epoch_start_marker = timer()
                    log_func.log_start_train(job_name, os.getpid(), assign_device)
                    logit.fit([train_input_ids, train_input_masks, train_segment_ids],
                              train_labels,
                              epochs=1,
                              batch_size=train_batchsize,
                              verbose=0)

                    # start evaluation phrase
                    log_func.log_start_eval(job_name, os.getpid(), assign_device)
                    scores = logit.evaluate([train_input_ids[0:offset],
                                             train_input_masks[0:offset],
                                             train_segment_ids[0:offset]],
                                            train_labels[0:offset],
                                            verbose=0)
                    cur_accuracy = scores[1]
                    log_func.log_end_eval(job_name, cur_accuracy, assign_device)

                    pre_accuracy = job_accuracy_dict[job_name]

                    epoch_end_marker = timer()

                    # tracking the time and accuracy
                    job_runtime_dict[job_name] += epoch_end_marker - epoch_start_marker
                    job_epoch_dict[job_name] += 1
                    job_accuracy_dict[job_name] = cur_accuracy

                    # add current results
                    dl_estimator.add_actual_data(job_key=job_name,
                                                 accuracy=cur_accuracy,
                                                 epoch=job_epoch_dict[job_name])

                    if job_slo == 'accuracy':
                        if job_accuracy_dict[job_name] >= job_slo_value:
                            end_time_overall = timer()
                            job_completion_time_dict[job_name] = end_time_overall - start_time_overall
                            job_attain_dict[job_name] = 1
                            log_func.log_time_accuracy(job_name,
                                                       cur_accuracy,
                                                       shared_runtime_history,
                                                       job_epoch_dict,
                                                       shared_accuracy_history)
                            logit.save(model_ckpt_save_path + '/' + job_name + '.h5')
                            msg_slot = 'job {} reaches the SLO'.format(job_id)
                            gpu_slot_rotary[gpu_device] = 0
                            sem_rotary.release()
                            return msg_slot

                        if job_epoch_dict[job_name] >= job_slo_max_time:
                            end_time_overall = timer()
                            job_completion_time_dict[job_name] = end_time_overall - start_time_overall
                            log_func.log_time_accuracy(job_name,
                                                       cur_accuracy,
                                                       shared_runtime_history,
                                                       job_epoch_dict,
                                                       shared_accuracy_history)
                            logit.save(model_ckpt_save_path + '/' + job_name + '.h5')
                            msg_slot = 'job {} is finished'.format(job_id)
                            gpu_slot_rotary[gpu_device] = 0
                            sem_rotary.release()
                            return msg_slot

                    elif job_slo == 'convergence':
                        delta = round(abs(cur_accuracy - pre_accuracy), 4)
                        if delta <= job_slo_value:
                            end_time_overall = timer()
                            job_completion_time_dict[job_name] = end_time_overall - start_time_overall
                            job_attain_dict[job_name] = 1
                            log_func.log_time_accuracy(job_name,
                                                       cur_accuracy,
                                                       shared_runtime_history,
                                                       job_epoch_dict,
                                                       shared_accuracy_history)
                            logit.save(model_ckpt_save_path + '/' + job_name + '.h5')
                            msg_slot = 'job {} reaches the SLO'.format(job_id)
                            gpu_slot_rotary[gpu_device] = 0
                            sem_rotary.release()
                            return msg_slot

                        if job_epoch_dict[job_name] >= job_slo_max_time:
                            end_time_overall = timer()
                            job_completion_time_dict[job_name] = end_time_overall - start_time_overall
                            log_func.log_time_accuracy(job_name,
                                                       cur_accuracy,
                                                       shared_runtime_history,
                                                       job_epoch_dict,
                                                       shared_accuracy_history)
                            logit.save(model_ckpt_save_path + '/' + job_name + '.h5')
                            msg_slot = 'job {} is finished'.format(job_id)
                            gpu_slot_rotary[gpu_device] = 0
                            sem_rotary.release()
                            return msg_slot

                    elif job_slo == 'runtime':
                        if job_epoch_dict[job_name] >= job_slo_value:
                            end_time_overall = timer()
                            job_completion_time_dict[job_name] = end_time_overall - start_time_overall
                            job_attain_dict[job_name] = 1
                            log_func.log_time_accuracy(job_name,
                                                       cur_accuracy,
                                                       shared_runtime_history,
                                                       job_epoch_dict,
                                                       shared_accuracy_history)
                            logit.save(model_ckpt_save_path + '/' + job_name + '.h5')
                            msg_slot = 'job {} reaches the SLO'.format(job_id)
                            gpu_slot_rotary[gpu_device] = 0
                            sem_rotary.release()
                            return msg_slot
                    else:
                        raise ValueError('the job objective type is not supported')

                    log_func.log_time_accuracy(job_name,
                                               cur_accuracy,
                                               shared_runtime_history,
                                               job_epoch_dict,
                                               shared_accuracy_history)

                    slot_end_marker = timer()
                    running_slot_time = slot_end_marker - slot_start_marker

                logit.save(model_ckpt_save_path + '/' + job_name + '.h5')

    elif job_model == 'lstm' or job_model == 'bilstm':
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
                                    opt=model_opt,
                                    lr=model_learn_rate)
            logit, _ = model.build(word2index, tag2index)

            # load the checkpoint if it exists
            model_ckpt_save_path = cfg_path.ckpt_save_path + '/' + job_name
            if os.path.exists(model_ckpt_save_path + '/' + job_name + '.h5'):
                logit.load_weights(model_ckpt_save_path + '/' + job_name + '.h5')

            with tf.Session(config=initialize_config()) as sess:
                sess.run(tf.global_variables_initializer())

                # add the prepare time for this process
                preparation_end_marker = timer()
                job_runtime_dict[job_name] += preparation_end_marker - preparation_start_marker

                # check if the total runtime is less than running_slot
                while running_slot_time < running_slot:
                    epoch_start_marker = timer()
                    log_func.log_start_train(job_name, os.getpid(), assign_device)
                    logit.fit(train_sentences_x,
                              udtb_reader.to_categorical(train_tags_y, len(tag2index)),
                              batch_size=train_batchsize,
                              epochs=1,
                              verbose=0)

                    # start evaluation phrase
                    log_func.log_start_eval(job_name, os.getpid(), assign_device)
                    scores = logit.evaluate(val_sentences_x,
                                            udtb_reader.to_categorical(val_tags_y, len(tag2index)),
                                            verbose=0)
                    cur_accuracy = scores[1]
                    log_func.log_end_eval(job_name, cur_accuracy, assign_device)

                    pre_accuracy = job_accuracy_dict[job_name]

                    epoch_end_marker = timer()

                    # tracking the time and accuracy
                    job_runtime_dict[job_name] += epoch_end_marker - epoch_start_marker
                    job_epoch_dict[job_name] += 1
                    job_accuracy_dict[job_name] = cur_accuracy

                    # add current results
                    dl_estimator.add_actual_data(job_key=job_name,
                                                 accuracy=cur_accuracy,
                                                 epoch=job_epoch_dict[job_name])

                    if job_slo == 'accuracy':
                        if job_accuracy_dict[job_name] >= job_slo_value:
                            end_time_overall = timer()
                            job_completion_time_dict[job_name] = end_time_overall - start_time_overall
                            job_attain_dict[job_name] = 1
                            log_func.log_time_accuracy(job_name,
                                                       cur_accuracy,
                                                       shared_runtime_history,
                                                       job_epoch_dict,
                                                       shared_accuracy_history)
                            logit.save(model_ckpt_save_path + '/' + job_name + '.h5')
                            msg_slot = 'job {} reaches the SLO'.format(job_id)
                            gpu_slot_rotary[gpu_device] = 0
                            sem_rotary.release()
                            return msg_slot

                        if job_epoch_dict[job_name] >= job_slo_max_time:
                            end_time_overall = timer()
                            job_completion_time_dict[job_name] = end_time_overall - start_time_overall
                            log_func.log_time_accuracy(job_name,
                                                       cur_accuracy,
                                                       shared_runtime_history,
                                                       job_epoch_dict,
                                                       shared_accuracy_history)
                            logit.save(model_ckpt_save_path + '/' + job_name + '.h5')
                            msg_slot = 'job {} is finished'.format(job_id)
                            gpu_slot_rotary[gpu_device] = 0
                            sem_rotary.release()
                            return msg_slot

                    elif job_slo == 'convergence':
                        delta = round(abs(cur_accuracy - pre_accuracy), 4)
                        if delta <= job_slo_value:
                            end_time_overall = timer()
                            job_completion_time_dict[job_name] = end_time_overall - start_time_overall
                            job_attain_dict[job_name] = 1
                            log_func.log_time_accuracy(job_name,
                                                       cur_accuracy,
                                                       shared_runtime_history,
                                                       job_epoch_dict,
                                                       shared_accuracy_history)
                            logit.save(model_ckpt_save_path + '/' + job_name + '.h5')
                            msg_slot = 'job {} reaches the SLO'.format(job_id)
                            gpu_slot_rotary[gpu_device] = 0
                            sem_rotary.release()
                            return msg_slot

                        if job_epoch_dict[job_name] >= job_slo_max_time:
                            end_time_overall = timer()
                            job_completion_time_dict[job_name] = end_time_overall - start_time_overall
                            log_func.log_time_accuracy(job_name,
                                                       cur_accuracy,
                                                       shared_runtime_history,
                                                       job_epoch_dict,
                                                       shared_accuracy_history)
                            logit.save(model_ckpt_save_path + '/' + job_name + '.h5')
                            msg_slot = 'job {} is finished'.format(job_id)
                            gpu_slot_rotary[gpu_device] = 0
                            sem_rotary.release()
                            return msg_slot

                    elif job_slo == 'runtime':
                        if job_epoch_dict[job_name] >= job_slo_value:
                            end_time_overall = timer()
                            job_completion_time_dict[job_name] = end_time_overall - start_time_overall
                            job_attain_dict[job_name] = 1
                            log_func.log_time_accuracy(job_name,
                                                       cur_accuracy,
                                                       shared_runtime_history,
                                                       job_epoch_dict,
                                                       shared_accuracy_history)
                            logit.save(model_ckpt_save_path + '/' + job_name + '.h5')
                            msg_slot = 'job {} reaches the SLO'.format(job_id)
                            gpu_slot_rotary[gpu_device] = 0
                            sem_rotary.release()
                            return msg_slot
                    else:
                        raise ValueError('the job objective type is not supported')

                    log_func.log_time_accuracy(job_name,
                                               cur_accuracy,
                                               shared_runtime_history,
                                               job_epoch_dict,
                                               shared_accuracy_history)

                    slot_end_marker = timer()
                    running_slot_time = slot_end_marker - slot_start_marker

            logit.save(model_ckpt_save_path + '/' + job_name + '.h5')

    else:
        img_w = 32
        img_h = 32
        num_chn = 3
        num_cls = 10

        # load cifar10 data
        train_feature, train_labels, eval_feature, eval_labels = cifar_reader.load_cifar10_keras()

        with tf.device(assign_device):
            feature_ph = tf.placeholder(tf.float32, [None, img_w, img_h, num_chn])
            label_ph = tf.placeholder(tf.int64, [None, num_cls])
            train_op, eval_op, _ = build_cv_model(job_data,
                                                  model_opt,
                                                  model_learn_rate,
                                                  num_cls,
                                                  feature_ph,
                                                  label_ph)

            # init the tf saver for checkpoint
            saver = tf.train.Saver()

            with tf.Session(config=initialize_config()) as sess:
                # check if the checkpoint file exist
                model_ckpt_save_path = cfg_path.ckpt_save_path + '/' + job_name
                checkpoint_file = model_ckpt_save_path + '/model_ckpt'
                if os.path.isfile(checkpoint_file + '.meta'):
                    saver.restore(sess, checkpoint_file)
                else:
                    sess.run(tf.global_variables_initializer())

                num_batch = train_labels.shape[0] // train_batchsize

                preparation_end_marker = timer()
                # add the prepare time for this process
                job_runtime_dict[job_name] += preparation_end_marker - preparation_start_marker

                # check if the total runtime is less than running_slot
                while running_slot_time < running_slot:
                    epoch_start_marker = timer()
                    log_func.log_start_train(job_name, os.getpid(), assign_device)
                    for b in range(num_batch):
                        batch_offset = b * train_batchsize
                        batch_end = (b + 1) * train_batchsize

                        train_data_batch = train_feature[batch_offset:batch_end]
                        train_label_batch = train_labels[batch_offset:batch_end]

                        sess.run(train_op, feed_dict={feature_ph: train_data_batch, label_ph: train_label_batch})

                    log_func.log_start_eval(job_name, os.getpid(), assign_device)
                    acc_sum = 0
                    eval_batch_size = 50
                    num_batch_eval = eval_labels.shape[0] // eval_batch_size
                    for be in range(num_batch_eval):
                        batch_offset = be * eval_batch_size
                        batch_end = (be + 1) * eval_batch_size
                        eval_feature_batch = eval_feature[batch_offset:batch_end]
                        eval_label_batch = eval_labels[batch_offset:batch_end]
                        acc_batch = sess.run(eval_op,
                                             feed_dict={feature_ph: eval_feature_batch, label_ph: eval_label_batch})
                        acc_sum += acc_batch

                    cur_accuracy = acc_sum / num_batch_eval
                    log_func.log_end_eval(job_name, cur_accuracy, assign_device)

                    pre_accuracy = job_accuracy_dict[job_name]

                    epoch_end_marker = timer()

                    # tracking the time and accuracy
                    job_runtime_dict[job_name] += epoch_end_marker - epoch_start_marker
                    job_epoch_dict[job_name] += 1
                    job_accuracy_dict[job_name] = cur_accuracy

                    # add current results
                    dl_estimator.add_actual_data(job_key=job_name,
                                                 accuracy=cur_accuracy,
                                                 epoch=job_epoch_dict[job_name])

                    if job_slo == 'accuracy':
                        if job_accuracy_dict[job_name] >= job_slo_value:
                            end_time_overall = timer()
                            job_completion_time_dict[job_name] = end_time_overall - start_time_overall
                            job_attain_dict[job_name] = 1
                            log_func.log_time_accuracy(job_name,
                                                       cur_accuracy,
                                                       shared_runtime_history,
                                                       job_epoch_dict,
                                                       shared_accuracy_history)
                            saver.save(sess, checkpoint_file)
                            msg_slot = 'job {} reaches the SLO'.format(job_id)
                            gpu_slot_rotary[gpu_device] = 0
                            sem_rotary.release()
                            return msg_slot

                        if job_epoch_dict[job_name] >= job_slo_max_time:
                            end_time_overall = timer()
                            job_completion_time_dict[job_name] = end_time_overall - start_time_overall
                            log_func.log_time_accuracy(job_name,
                                                       cur_accuracy,
                                                       shared_runtime_history,
                                                       job_epoch_dict,
                                                       shared_accuracy_history)
                            saver.save(sess, checkpoint_file)
                            msg_slot = 'job {} is finished'.format(job_id)
                            gpu_slot_rotary[gpu_device] = 0
                            sem_rotary.release()
                            return msg_slot

                    elif job_slo == 'convergence':
                        delta = round(abs(cur_accuracy - pre_accuracy), 4)
                        if delta <= job_slo_value:
                            end_time_overall = timer()
                            job_completion_time_dict[job_name] = end_time_overall - start_time_overall
                            job_attain_dict[job_name] = 1
                            log_func.log_time_accuracy(job_name,
                                                       cur_accuracy,
                                                       shared_runtime_history,
                                                       job_epoch_dict,
                                                       shared_accuracy_history)
                            saver.save(sess, checkpoint_file)
                            msg_slot = 'job {} reaches the SLO'.format(job_id)
                            gpu_slot_rotary[gpu_device] = 0
                            sem_rotary.release()
                            return msg_slot

                        if job_epoch_dict[job_name] >= job_slo_max_time:
                            end_time_overall = timer()
                            job_completion_time_dict[job_name] = end_time_overall - start_time_overall
                            log_func.log_time_accuracy(job_name,
                                                       cur_accuracy,
                                                       shared_runtime_history,
                                                       job_epoch_dict,
                                                       shared_accuracy_history)
                            saver.save(sess, checkpoint_file)
                            msg_slot = 'job {} is finished'.format(job_id)
                            gpu_slot_rotary[gpu_device] = 0
                            sem_rotary.release()
                            return msg_slot

                    elif job_slo == 'runtime':
                        if job_epoch_dict[job_name] >= job_slo_value:
                            end_time_overall = timer()
                            job_completion_time_dict[job_name] = end_time_overall - start_time_overall
                            job_attain_dict[job_name] = 1
                            log_func.log_time_accuracy(job_name,
                                                       cur_accuracy,
                                                       shared_runtime_history,
                                                       job_epoch_dict,
                                                       shared_accuracy_history)
                            saver.save(sess, checkpoint_file)
                            msg_slot = 'job {} reaches the SLO'.format(job_id)
                            gpu_slot_rotary[gpu_device] = 0
                            sem_rotary.release()
                            return msg_slot
                    else:
                        raise ValueError('the job objective type is not supported')

                    log_func.log_time_accuracy(job_name,
                                               cur_accuracy,
                                               shared_runtime_history,
                                               job_epoch_dict,
                                               shared_accuracy_history)

                    slot_end_marker = timer()
                    running_slot_time = slot_end_marker - slot_start_marker

                saver.save(sess, checkpoint_file)

    # exceed the running slot and haven't achieve goal so put the job back to the queue
    job_list_rotary.append(job_data)
    job_queue_anony.put(job_anony)
    msg_slot = 'job {} is finished the current running slot'.format(job_id)

    gpu_slot_rotary[gpu_device] = 0
    sem_rotary.release()
    return msg_slot


if __name__ == "__main__":
    proc_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print('============= the whole exp starts at: {}===================='.format(proc_start_time))

    # start timing for job completion
    start_time_overall = timer()

    # create folder if not exist
    if not os.path.exists(cfg_path.ckpt_save_path):
        os.makedirs(cfg_path.ckpt_save_path)

    #######################################################
    # get parameters from configuration
    #######################################################

    num_gpu = cfg_rotary.num_gpu
    running_slot = cfg_rotary.running_slot

    assert cfg_rotary.cv_light_ratio + cfg_rotary.cv_med_ratio + cfg_rotary.cv_heavy_ratio == 1

    #######################################################
    # generate the workload
    #######################################################

    wg = WorkloadGenerator(cfg_rotary.dlt_workload_size,
                           cfg_rotary.dlt_cv_light_ratio,
                           cfg_rotary.dlt_cv_med_ratio,
                           cfg_rotary.dlt_cv_heavy_ratio,
                           cfg_rotary.dlt_nlp_light_ratio,
                           cfg_rotary.dlt_nlp_med_ratio,
                           cfg_rotary.dlt_nlp_heavy_ratio,
                           cfg_rotary.convergence_ratio,
                           cfg_rotary.accuracy_ratio,
                           cfg_rotary.runtime_ratio,
                           cfg_rotary.random_seed)

    ml_workload = wg.generate_workload()

    for i in ml_workload:
        print(i)

    #######################################################
    # prepare everything necessary
    #######################################################

    # queue for trial phrase
    job_queue_trial = mp.Manager().Queue()

    # queue for checking if all jobs are completed
    job_queue_anony = mp.Manager().Queue()

    # list for rotary phrase
    job_list_rotary = mp.Manager().list()

    # list for getting the total parameters of each job
    ml_workload_shared = mp.Manager().list()

    # dict for storing job's current accuracy
    job_accuracy_dict = mp.Manager().dict()
    # dict for storing job's overall running time
    job_runtime_dict = mp.Manager().dict()
    # dict for storing job's overall training epochs
    job_epoch_dict = mp.Manager().dict()
    # dict for storing parameters of all jobs
    job_parameters_dict = mp.Manager().dict()
    # dict for storing job's completion time (achieving SLO or being terminated)
    job_completion_time_dict = mp.Manager().dict()
    # dict for storing if the job achieving the SLO
    job_attain_dict = mp.Manager().dict()
    # dict for storing the time of a single training epoch of each job
    job_epochtime_dict = mp.Manager().dict()

    # dict for storing job's progress
    job_progress_dict = dict()
    # dict for tracking epochs with wall-time
    job_runtime_history = dict()
    # dict for tracking accuracy with wall-time
    job_accuracy_history = dict()

    # init the estimator
    # read all the model-accuracy files
    dl_estimator = DLEstimator(topk=5)
    for f in os.listdir('./knowledgebase'):
        model_acc_file = os.getcwd() + '/knowledgebase/' + f
        dl_estimator.import_accuracy_dataset(model_acc_file)

    dl_estimator.import_workload(ml_workload)

    # init process pool
    proc_pool_trial = mp.Pool(num_gpu, maxtasksperchild=1)
    proc_pool_rotary = mp.Pool(num_gpu, maxtasksperchild=1)

    # init some dicts to track the progress
    for job in ml_workload:
        job_key = str(job['id']) + '-' + job['model']
        job_queue_trial.put(job)
        job_list_rotary.append(job)
        job_queue_anony.put(job_key)
        ml_workload_shared.append(job)

        job_progress_dict[job_key] = 0
        job_accuracy_dict[job_key] = 0
        job_runtime_dict[job_key] = 0
        job_epoch_dict[job_key] = 0
        job_epochtime_dict[job_key] = 0
        job_parameters_dict[job_key] = 0
        job_completion_time_dict[job_key] = 0
        job_attain_dict[job_key] = 0

        # init the dict for tracking
        init_sub_accuracy_list = mp.Manager().list()
        init_sub_accuracy_list.append('0:' + proc_start_time)
        job_accuracy_history[job_key] = init_sub_accuracy_list
        init_sub_runtime_list = mp.Manager().list()
        init_sub_runtime_list.append('0:' + proc_start_time)
        job_runtime_history[job_key] = init_sub_runtime_list

    #######################################################
    # start the trial process
    #######################################################

    results_trial = list()
    for idx in range(len(ml_workload)):
        result = proc_pool_trial.apply_async(train_job_trial, args=(job_runtime_history,
                                                                    job_accuracy_history))
        results_trial.append(result)

    for i in results_trial:
        i.wait()

    for i in results_trial:
        if i.ready():
            if i.successful():
                print(i.get())

    for key in job_accuracy_dict:
        print(key, '[accuracy]->', job_accuracy_dict[key])

    for key in job_runtime_dict:
        print(key, '[time]->', job_runtime_dict[key])

    for key in job_epoch_dict:
        print(key, '[epoch]->', job_epoch_dict[key])

    for key in job_epochtime_dict:
        print(key, '[epoch_time]->', job_epochtime_dict[key])

    for ml_job in ml_workload_shared:
        print('Job {}: {} parameters'.format(ml_job['id'], ml_job['num_parameters']))

    #######################################################
    # prepare estimation for the workload
    #######################################################

    dl_estimator.prepare_workload(ml_workload_shared)

    #######################################################
    # start the rotary process
    #######################################################

    fairness = True
    while not job_queue_anony.empty():
        results_rotary = list()
        job_select = job_list_rotary[0]
        print('************* current rotary queue: {} *************'.format(job_queue_anony.qsize()))
        for gpu_idx in range(num_gpu):
            r_score_mark = float('inf') if fairness else float('-inf')
            r_score = 0

            for job_ins in job_list_rotary:
                job_ins_key = str(job_ins['id']) + '-' + job_ins['model']

                job_ins_slo = job_ins['goal_type']
                job_ins_slo_value = job_ins['goal_value']

                if job_ins_slo == 'runtime':
                    current_epoch = job_epoch_dict[job_ins_key]
                    r_score = current_epoch / job_ins_slo_value
                    job_progress_dict[job_ins_key] = r_score

                elif job_ins_slo == 'accuracy':
                    job_ins_slo_max_time = job_ins['goal_value_extra']
                    current_epoch = job_epoch_dict[job_ins_key]
                    estimate_all_epoch = round(dl_estimator.predict_epoch(job_ins, job_ins_slo_value))
                    if estimate_all_epoch > job_ins_slo_max_time:
                        r_score = current_epoch / job_ins_slo_max_time
                    else:
                        r_score = current_epoch / estimate_all_epoch if estimate_all_epoch != 0 else 0
                    job_progress_dict[job_ins_key] = r_score

                elif job_ins_slo == 'convergence':
                    job_ins_slo_max_time = job_ins['goal_value_extra']
                    current_epoch = job_epoch_dict[job_ins_key]
                    current_accuracy = job_accuracy_dict[job_ins_key]
                    expected_accuracy = current_accuracy + job_ins_slo_value
                    estimate_all_epoch = round(dl_estimator.predict_epoch(job_ins, expected_accuracy))
                    if estimate_all_epoch <= 0:
                        r_score = 1
                    else:
                        if estimate_all_epoch > job_ins_slo_max_time:
                            r_score = current_epoch / job_ins_slo_max_time
                        else:
                            r_score = current_epoch / estimate_all_epoch

                    job_progress_dict[job_ins_key] = r_score

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
            '''
            print('------------------------------------------------')
            for job in job_list_rotary:
                job_key = str(job['id']) + '-' + job['model']
                print('Job {}:, R-Score:{}'.format(job_key, job_progress_dict[job_key]))
            print('------------------------------------------------')
            '''
            print('$$$$$ JOB SELECTION: {} $$$$$'.format(job_select))

            try:
                job_list_rotary.remove(job_select)
            except ValueError:
                msg_err = 'job has been handled by other GPU'
                print(msg_err)
                continue

            result = proc_pool_rotary.apply_async(train_job, args=(job_select,
                                                                   job_runtime_history,
                                                                   job_accuracy_history))
            results_rotary.append(result)

        for i in results_rotary:
            i.wait()

        for i in results_rotary:
            if i.ready():
                if i.successful():
                    print(i.get())

        fairness = False
        threshold = 0.5
        for job in job_list_rotary:
            job_key = str(job['id']) + '-' + job['model']
            if job_progress_dict[job_key] < threshold:
                fairness = True
                break

        if fairness:
            print('||||||||||||||||||| USING FAIRNESS POLICY |||||||||||||||||||')
        else:
            print('||||||||||||||||||| USING AGGRESSIVE POLICY |||||||||||||||||||')

    #######################################################
    # printout the log information
    #######################################################

    for key in job_accuracy_dict:
        print('{} [accuracy]-> {}'.format(key, job_accuracy_dict[key]))

    for key in job_runtime_dict:
        print('{} [runtime]-> {}'.format(key, job_runtime_dict[key]))

    for key in job_epoch_dict:
        print('{} [epoch]-> {}'.format(key, job_epoch_dict[key]))

    for key in job_completion_time_dict:
        print('{} [completion time]-> {}'.format(key, job_completion_time_dict[key]))

    for key in job_attain_dict:
        print('{} [attainment]-> {}'.format(key, job_attain_dict[key]))

    print("show the history")

    for key in job_accuracy_history:
        print('{} [acc_history]-> {}'.format(key, job_accuracy_history[key]))

    for key in job_runtime_history:
        print('{} [runtime_history]-> {}'.format(key, job_runtime_history[key]))

    proc_end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print('============= the whole exp finish at: {}===================='.format(proc_end_time))

    # print('total log count: {}'.format(log_func.log_counter.value))
    # print('log time: {}'.format(log_func.log_time.value))
    # print('rotary time: {}'.format(log_func.rotary_time.value))
