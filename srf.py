import tensorflow as tf
import multiprocessing as mp
from timeit import default_timer as timer
import os
import queue
import numpy as np
from datetime import datetime

import config.config_rotary as cfg_rotary
import config.config_path as cfg_path
import workload.tensorflow_cifar.tools.cifar_reader as cifar_reader
import workload.tensorflow_nlp.tools.udtb_reader as udtb_reader
import workload.tensorflow_nlp.tools.lmrd_reader as lmrd_reader
from workload.workload_generator import WorkloadGenerator
from utils.model_tool import build_cv_model, build_nlp_model
from utils.log_func import log_time_accuracy
from utils.tf_func import initialize_config, initialize_vars


def compared_item(item):
    return item['goal_value']


def train_job_runtime(gpu_id,
                      shared_runtime_history,
                      shared_accuracy_history):
    preparation_start_marker = timer()

    ml_workload_runtime.sort(key=compared_item, reverse=True)
    try:
        job_data = ml_workload_runtime.pop()
    except IndexError:
        return

    job_name = str(job_data['id']) + '-' + job_data['model']
    print('get job {}'.format(job_name))

    # get the device id
    assign_device = '/gpu:' + str(gpu_id)
    print('running on device {}'.format(assign_device))

    # get opt, learning rate, batch size and image size for training
    model_opt = job_data['opt']
    model_learn_rate = job_data['learn_rate']
    train_batchsize = job_data['batch_size']

    if job_data['model'] == 'bert':
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

                model = build_nlp_model(model_type=job_data['model'],
                                        max_length=128,
                                        opt=model_opt,
                                        lr=model_learn_rate)

                logit, _ = model.build()

                # load the checkpoint if it exists
                if os.path.exists(model_ckpt_save_path + '/' + job_name + '.h5'):
                    logit.load_weights(model_ckpt_save_path + '/' + job_name + '.h5')

                # Instantiate variables
                initialize_vars(sess)

                # add the preparation time for this process
                preparation_end_marker = timer()
                job_runtime_dict[job_name] += preparation_end_marker - preparation_start_marker

                # check if the total runtime is less than running_slot
                while True:
                    epoch_start_marker = timer()

                    logit.fit([train_input_ids, train_input_masks, train_segment_ids],
                              train_labels,
                              epochs=1,
                              batch_size=train_batchsize,
                              verbose=0)

                    # start evaluation phrase
                    print('start evaluating job {} at process {}'.format(job_name, os.getpid()))
                    scores = logit.evaluate([train_input_ids[0:offset],
                                             train_input_masks[0:offset],
                                             train_segment_ids[0:offset]],
                                            train_labels[0:offset],
                                            verbose=0)
                    cur_accuracy = scores[1]
                    print('job {} evaluation accuracy:{}'.format(job_name, cur_accuracy))

                    epoch_end_marker = timer()

                    # tracking time and accuracy
                    job_runtime_dict[job_name] += epoch_end_marker - epoch_start_marker
                    job_epoch_dict[job_name] += 1
                    job_accuracy_dict[job_name] = cur_accuracy

                    # decision phrase
                    if job_epoch_dict[job_name] >= job_data['goal_value']:
                        end_time_overall = timer()
                        job_completion_time_dict[job_name] = end_time_overall - start_time_overall
                        job_attain_dict[job_name] = 1
                        log_time_accuracy(job_name,
                                          cur_accuracy,
                                          shared_runtime_history,
                                          job_epoch_dict,
                                          shared_accuracy_history)
                        # save the model as the job achieves SLO and exit the current process
                        logit.save(model_ckpt_save_path + '/' + job_name + '.h5')
                        msg = 'job {} reaches SLO'.format(job_data['id'])
                        return msg

                    log_time_accuracy(job_name,
                                      cur_accuracy,
                                      shared_runtime_history,
                                      job_epoch_dict,
                                      shared_accuracy_history)

    elif job_data['model'] == 'lstm' or job_data['model'] == 'bilstm':
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
            model = build_nlp_model(model_type=job_data['model'],
                                    max_length=MAX_LENGTH,
                                    opt=model_opt,
                                    lr=model_learn_rate)

            logit, _ = model.build(word2index, tag2index)

            # load the checkpoint if it exists
            if os.path.exists(model_ckpt_save_path + '/' + job_name + '.h5'):
                logit.load_weights(model_ckpt_save_path + '/' + job_name + '.h5')

            with tf.Session(config=initialize_config()) as sess:
                sess.run(tf.global_variables_initializer())

                # add the preparation time for this process
                preparation_end_marker = timer()
                job_runtime_dict[job_name] += preparation_end_marker - preparation_start_marker

                # check if the total runtime is less than running_slot
                while True:
                    epoch_start_marker = timer()

                    logit.fit(train_sentences_x,
                              udtb_reader.to_categorical(train_tags_y, len(tag2index)),
                              batch_size=train_batchsize,
                              epochs=1,
                              verbose=0)

                    # start evaluation phrase
                    print('start evaluating job {} at process {}'.format(job_name, os.getpid()))
                    scores = logit.evaluate(val_sentences_x,
                                            udtb_reader.to_categorical(val_tags_y, len(tag2index)),
                                            verbose=0)
                    cur_accuracy = scores[1]
                    print('job {} evaluation accuracy:{}'.format(job_name, cur_accuracy))

                    epoch_end_marker = timer()

                    # tracking time and accuracy
                    job_runtime_dict[job_name] += epoch_end_marker - epoch_start_marker
                    job_epoch_dict[job_name] += 1
                    job_accuracy_dict[job_name] = cur_accuracy

                    # decision phrase
                    if job_epoch_dict[job_name] >= job_data['goal_value']:
                        end_time_overall = timer()
                        job_completion_time_dict[job_name] = end_time_overall - start_time_overall
                        job_attain_dict[job_name] = 1
                        log_time_accuracy(job_name,
                                          cur_accuracy,
                                          shared_runtime_history,
                                          job_epoch_dict,
                                          shared_accuracy_history)
                        # save the model as the job achieves SLO and exit the current process
                        logit.save(model_ckpt_save_path + '/' + job_name + '.h5')
                        msg = 'job {} reaches SLO'.format(job_data['id'])
                        return msg

                    log_time_accuracy(job_name,
                                      cur_accuracy,
                                      shared_runtime_history,
                                      job_epoch_dict,
                                      shared_accuracy_history)
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
            train_op, eval_op, total_parameters = build_cv_model(job_data,
                                                                 model_opt,
                                                                 model_learn_rate,
                                                                 num_cls,
                                                                 feature_ph,
                                                                 label_ph)

            # init the tf saver for checkpoint
            saver = tf.train.Saver()

            # get the path of checkpoint
            model_ckpt_save_path = cfg_path.ckpt_save_path + '/' + job_name

            if not os.path.exists(model_ckpt_save_path):
                os.makedirs(model_ckpt_save_path)

            with tf.Session(config=initialize_config()) as sess:
                # check if the checkpoint file exist
                checkpoint_file = model_ckpt_save_path + '/model_ckpt'
                if os.path.isfile(checkpoint_file + '.meta'):
                    saver.restore(sess, checkpoint_file)
                else:
                    sess.run(tf.global_variables_initializer())

                num_batch = train_labels.shape[0] // train_batchsize

                # add the preparation time for this process
                preparation_end_marker = timer()
                job_runtime_dict[job_name] += preparation_end_marker - preparation_start_marker

                while True:
                    epoch_start_marker = timer()

                    for b in range(num_batch):
                        # print('step {} / {}'.format(i + 1, num_batch))
                        batch_offset = b * train_batchsize
                        batch_end = (b + 1) * train_batchsize

                        train_data_batch = train_feature[batch_offset:batch_end]
                        train_label_batch = train_labels[batch_offset:batch_end]

                        sess.run(train_op, feed_dict={feature_ph: train_data_batch, label_ph: train_label_batch})

                    print('start evaluation phrase')
                    acc_sum = 0
                    eval_batch_size = 50
                    num_batch_eval = eval_labels.shape[0] // eval_batch_size
                    for be in range(num_batch_eval):
                        # print('evaluation step %d / %d' % (i + 1, num_batch_eval))
                        batch_offset = be * eval_batch_size
                        batch_end = (be + 1) * eval_batch_size
                        eval_feature_batch = eval_feature[batch_offset:batch_end]
                        eval_label_batch = eval_labels[batch_offset:batch_end]
                        acc_batch = sess.run(eval_op,
                                             feed_dict={feature_ph: eval_feature_batch, label_ph: eval_label_batch})
                        acc_sum += acc_batch

                    cur_accuracy = acc_sum / num_batch_eval
                    print('job {} evaluation accuracy:{}'.format(job_name, cur_accuracy))

                    epoch_end_marker = timer()

                    # tracking time and accuracy
                    job_runtime_dict[job_name] += epoch_end_marker - epoch_start_marker
                    job_epoch_dict[job_name] += 1
                    job_accuracy_dict[job_name] = cur_accuracy

                    if job_epoch_dict[job_name] >= job_data['goal_value']:
                        end_time_overall = timer()
                        job_completion_time_dict[job_name] = end_time_overall - start_time_overall
                        job_attain_dict[job_name] = 1
                        log_time_accuracy(job_name,
                                          cur_accuracy,
                                          shared_runtime_history,
                                          job_epoch_dict,
                                          shared_accuracy_history)
                        # save the model as the job achieves SLO and exit the current process
                        saver.save(sess, checkpoint_file)
                        msg = 'job {} reaches SLO'.format(job_data['id'])
                        return msg

                    log_time_accuracy(job_name,
                                      cur_accuracy,
                                      shared_runtime_history,
                                      job_epoch_dict,
                                      shared_accuracy_history)


def train_job_others(gpu_id,
                     shared_runtime_history,
                     shared_accuracy_history):
    preparation_start_marker = timer()
    slot_start_marker = timer()

    # count the training time of this slot
    running_slot_time = 0

    # get the job data from the queue
    try:
        job_data = job_queue_others.get_nowait()
    except queue.Empty:
        return

    job_name = str(job_data['id']) + '-' + job_data['model']
    print('get job {}'.format(job_name))

    # get the device id
    assign_device = '/gpu:' + str(gpu_id)
    print('running on device {}'.format(assign_device))

    # get opt, learning rate, batch size and image size for training
    model_opt = job_data['opt']
    model_learn_rate = job_data['learn_rate']
    train_batchsize = job_data['batch_size']

    if job_data['model'] == 'bert':
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

                model = build_nlp_model(model_type=job_data['model'],
                                        max_length=128,
                                        opt=model_opt,
                                        lr=model_learn_rate)
                logit, _ = model.build()

                # load the checkpoint if it exists
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

                    logit.fit([train_input_ids, train_input_masks, train_segment_ids],
                              train_labels,
                              epochs=1,
                              batch_size=train_batchsize,
                              verbose=0)

                    # start evaluation phrase
                    print('start evaluating job {} at process {}'.format(job_name, os.getpid()))
                    scores = logit.evaluate([train_input_ids[0:offset],
                                             train_input_masks[0:offset],
                                             train_segment_ids[0:offset]],
                                            train_labels[0:offset],
                                            verbose=0)
                    cur_accuracy = scores[1]
                    print('job {} evaluation accuracy:{}'.format(job_name, cur_accuracy))

                    pre_accuracy = job_accuracy_dict[job_name]

                    epoch_end_marker = timer()

                    # tracking the time and accuracy
                    job_runtime_dict[job_name] += epoch_end_marker - epoch_start_marker
                    job_epoch_dict[job_name] += 1
                    job_accuracy_dict[job_name] = cur_accuracy

                    # decision phrase
                    if job_data['goal_type'] == 'accuracy':
                        if job_accuracy_dict[job_name] >= job_data['goal_value']:
                            end_time_overall = timer()
                            job_completion_time_dict[job_name] = end_time_overall - start_time_overall
                            job_attain_dict[job_name] = 1
                            log_time_accuracy(job_name,
                                              cur_accuracy,
                                              shared_runtime_history,
                                              job_epoch_dict,
                                              shared_accuracy_history)
                            logit.save(model_ckpt_save_path + '/' + job_name + '.h5')
                            msg = 'job {} reaches the SLO'.format(job_data['id'])
                            return msg

                        if job_epoch_dict[job_name] >= job_data['goal_value_extra']:
                            end_time_overall = timer()
                            job_completion_time_dict[job_name] = end_time_overall - start_time_overall
                            log_time_accuracy(job_name,
                                              cur_accuracy,
                                              shared_runtime_history,
                                              job_epoch_dict,
                                              shared_accuracy_history)
                            logit.save(model_ckpt_save_path + '/' + job_name + '.h5')
                            msg = 'job {} is finished'.format(job_data['id'])
                            return msg

                    elif job_data['goal_type'] == 'convergence':
                        delta = round(abs(cur_accuracy - pre_accuracy), 4)
                        if delta <= job_data['goal_value']:
                            end_time_overall = timer()
                            job_completion_time_dict[job_name] = end_time_overall - start_time_overall
                            job_attain_dict[job_name] = 1
                            log_time_accuracy(job_name,
                                              cur_accuracy,
                                              shared_runtime_history,
                                              job_epoch_dict,
                                              shared_accuracy_history)
                            logit.save(model_ckpt_save_path + '/' + job_name + '.h5')
                            msg = 'job {} reaches the SLO'.format(job_data['id'])
                            return msg

                        if job_epoch_dict[job_name] >= job_data['goal_value_extra']:
                            end_time_overall = timer()
                            job_completion_time_dict[job_name] = end_time_overall - start_time_overall
                            log_time_accuracy(job_name,
                                              cur_accuracy,
                                              shared_runtime_history,
                                              job_epoch_dict,
                                              shared_accuracy_history)
                            logit.save(model_ckpt_save_path + '/' + job_name + '.h5')
                            msg = 'job {} is finished'.format(job_data['id'])
                            return msg
                    else:
                        raise ValueError('the job objective type is not supported')

                    log_time_accuracy(job_name,
                                      cur_accuracy,
                                      shared_runtime_history,
                                      job_epoch_dict,
                                      shared_accuracy_history)

                    slot_end_marker = timer()
                    running_slot_time = slot_end_marker - slot_start_marker

                # save the model/job since the job has run for the current slot but doesn't achieve SLO
                logit.save(model_ckpt_save_path + '/' + job_name + '.h5')

    elif job_data['model'] == 'lstm' or job_data['model'] == 'bilstm':
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
            model = build_nlp_model(model_type=job_data['model'],
                                    max_length=MAX_LENGTH,
                                    opt=model_opt,
                                    lr=model_learn_rate)

            logit, total_parameters = model.build(word2index, tag2index)

            # load the checkpoint if it exists
            if os.path.exists(model_ckpt_save_path + '/' + job_name + '.h5'):
                logit.load_weights(model_ckpt_save_path + '/' + job_name + '.h5')

            with tf.Session(config=initialize_config()) as sess:
                sess.run(tf.global_variables_initializer())

                # add the preparation time for this process
                preparation_end_marker = timer()
                job_runtime_dict[job_name] += preparation_end_marker - preparation_start_marker

                # check if the total runtime is less than running_slot
                while running_slot_time < running_slot:
                    epoch_start_marker = timer()

                    logit.fit(train_sentences_x,
                              udtb_reader.to_categorical(train_tags_y, len(tag2index)),
                              batch_size=train_batchsize,
                              epochs=1,
                              verbose=0)

                    # start evaluation phrase
                    print('start evaluating job {} at process {}'.format(job_name, os.getpid()))
                    scores = logit.evaluate(val_sentences_x,
                                            udtb_reader.to_categorical(val_tags_y, len(tag2index)),
                                            verbose=0)
                    cur_accuracy = scores[1]
                    print('job {} evaluation accuracy:{}'.format(job_name, cur_accuracy))

                    pre_accuracy = job_accuracy_dict[job_name]

                    epoch_end_marker = timer()

                    # tracking the time and accuracy
                    job_runtime_dict[job_name] += epoch_end_marker - epoch_start_marker
                    job_epoch_dict[job_name] += 1
                    job_accuracy_dict[job_name] = cur_accuracy

                    # decision phrase
                    if job_data['goal_type'] == 'accuracy':
                        if job_accuracy_dict[job_name] >= job_data['goal_value']:
                            end_time_overall = timer()
                            job_completion_time_dict[job_name] = end_time_overall - start_time_overall
                            job_attain_dict[job_name] = 1
                            log_time_accuracy(job_name,
                                              cur_accuracy,
                                              shared_runtime_history,
                                              job_epoch_dict,
                                              shared_accuracy_history)
                            logit.save(model_ckpt_save_path + '/' + job_name + '.h5')
                            msg = 'job {} reaches the SLO'.format(job_data['id'])
                            return msg

                        if job_epoch_dict[job_name] >= job_data['goal_value_extra']:
                            end_time_overall = timer()
                            job_completion_time_dict[job_name] = end_time_overall - start_time_overall
                            log_time_accuracy(job_name,
                                              cur_accuracy,
                                              shared_runtime_history,
                                              job_epoch_dict,
                                              shared_accuracy_history)
                            logit.save(model_ckpt_save_path + '/' + job_name + '.h5')
                            msg = 'job {} is finished'.format(job_data['id'])
                            return msg

                    elif job_data['goal_type'] == 'convergence':
                        delta = round(abs(cur_accuracy - pre_accuracy), 4)
                        if delta <= job_data['goal_value']:
                            end_time_overall = timer()
                            job_completion_time_dict[job_name] = end_time_overall - start_time_overall
                            job_attain_dict[job_name] = 1
                            log_time_accuracy(job_name,
                                              cur_accuracy,
                                              shared_runtime_history,
                                              job_epoch_dict,
                                              shared_accuracy_history)
                            logit.save(model_ckpt_save_path + '/' + job_name + '.h5')
                            msg = 'job {} reaches the SLO'.format(job_data['id'])
                            return msg

                        if job_epoch_dict[job_name] >= job_data['goal_value_extra']:
                            end_time_overall = timer()
                            job_completion_time_dict[job_name] = end_time_overall - start_time_overall
                            log_time_accuracy(job_name,
                                              cur_accuracy,
                                              shared_runtime_history,
                                              job_epoch_dict,
                                              shared_accuracy_history)
                            logit.save(model_ckpt_save_path + '/' + job_name + '.h5')
                            msg = 'job {} is finished'.format(job_data['id'])
                            return msg
                    else:
                        raise ValueError('the job objective type is not supported')

                    log_time_accuracy(job_name,
                                      cur_accuracy,
                                      shared_runtime_history,
                                      job_epoch_dict,
                                      shared_accuracy_history)

                    slot_end_marker = timer()
                    running_slot_time = slot_end_marker - slot_start_marker

                # save the model/job since the job has run for the current slot but doesn't achieve SLO
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
            train_op, eval_op, total_parameters = build_cv_model(job_data,
                                                                 model_opt,
                                                                 model_learn_rate,
                                                                 num_cls,
                                                                 feature_ph,
                                                                 label_ph)

            # init the tf saver for checkpoint
            saver = tf.train.Saver()

            # get the path of checkpoint
            model_ckpt_save_path = cfg_path.ckpt_save_path + '/' + job_name

            if not os.path.exists(model_ckpt_save_path):
                os.makedirs(model_ckpt_save_path)

            with tf.Session(config=initialize_config()) as sess:
                # check if the checkpoint file exist
                checkpoint_file = model_ckpt_save_path + '/model_ckpt'
                if os.path.isfile(checkpoint_file + '.meta'):
                    saver.restore(sess, checkpoint_file)
                else:
                    sess.run(tf.global_variables_initializer())

                num_batch = train_labels.shape[0] // train_batchsize

                # add the prepare time for this process
                preparation_end_marker = timer()
                job_runtime_dict[job_name] += preparation_end_marker - preparation_start_marker

                # check if the total runtime is less than running_slot
                while running_slot_time < running_slot:
                    epoch_start_marker = timer()

                    for b in range(num_batch):
                        # print('step {} / {}'.format(i + 1, num_batch))
                        batch_offset = b * train_batchsize
                        batch_end = (b + 1) * train_batchsize

                        train_data_batch = train_feature[batch_offset:batch_end]
                        train_label_batch = train_labels[batch_offset:batch_end]

                        sess.run(train_op, feed_dict={feature_ph: train_data_batch, label_ph: train_label_batch})

                    print('start evaluation phrase')
                    acc_sum = 0
                    eval_batch_size = 50
                    num_batch_eval = eval_labels.shape[0] // eval_batch_size
                    for be in range(num_batch_eval):
                        # print('evaluation step %d / %d' % (i + 1, num_batch_eval))
                        batch_offset = be * eval_batch_size
                        batch_end = (be + 1) * eval_batch_size
                        eval_feature_batch = eval_feature[batch_offset:batch_end]
                        eval_label_batch = eval_labels[batch_offset:batch_end]
                        acc_batch = sess.run(eval_op,
                                             feed_dict={feature_ph: eval_feature_batch, label_ph: eval_label_batch})
                        acc_sum += acc_batch

                    cur_accuracy = acc_sum / num_batch_eval
                    print('job {} evaluation accuracy:{}'.format(job_name, cur_accuracy))

                    pre_accuracy = job_accuracy_dict[job_name]

                    epoch_end_marker = timer()

                    # tracking the time and accuracy
                    job_runtime_dict[job_name] += epoch_end_marker - epoch_start_marker
                    job_epoch_dict[job_name] += 1
                    job_accuracy_dict[job_name] = cur_accuracy

                    # decision phrase
                    if job_data['goal_type'] == 'accuracy':
                        if job_accuracy_dict[job_name] >= job_data['goal_value']:
                            end_time_overall = timer()
                            job_completion_time_dict[job_name] = end_time_overall - start_time_overall
                            job_attain_dict[job_name] = 1
                            log_time_accuracy(job_name,
                                              cur_accuracy,
                                              shared_runtime_history,
                                              job_epoch_dict,
                                              shared_accuracy_history)
                            saver.save(sess, checkpoint_file)
                            msg = 'job {} reaches the SLO'.format(job_data['id'])
                            return msg

                        if job_epoch_dict[job_name] >= job_data['goal_value_extra']:
                            end_time_overall = timer()
                            job_completion_time_dict[job_name] = end_time_overall - start_time_overall
                            log_time_accuracy(job_name,
                                              cur_accuracy,
                                              shared_runtime_history,
                                              job_epoch_dict,
                                              shared_accuracy_history)
                            saver.save(sess, checkpoint_file)
                            msg = 'job {} is finished'.format(job_data['id'])
                            return msg

                    elif job_data['goal_type'] == 'convergence':
                        delta = round(abs(cur_accuracy - pre_accuracy), 4)
                        if delta <= job_data['goal_value']:
                            end_time_overall = timer()
                            job_completion_time_dict[job_name] = end_time_overall - start_time_overall
                            job_attain_dict[job_name] = 1
                            log_time_accuracy(job_name,
                                              cur_accuracy,
                                              shared_runtime_history,
                                              job_epoch_dict,
                                              shared_accuracy_history)
                            saver.save(sess, checkpoint_file)
                            msg = 'job {} reaches the SLO'.format(job_data['id'])
                            return msg

                        if job_epoch_dict[job_name] >= job_data['goal_value_extra']:
                            end_time_overall = timer()
                            job_completion_time_dict[job_name] = end_time_overall - start_time_overall
                            log_time_accuracy(job_name,
                                              cur_accuracy,
                                              shared_runtime_history,
                                              job_epoch_dict,
                                              shared_accuracy_history)
                            saver.save(sess, checkpoint_file)
                            msg = 'job {} is finished'.format(job_data['id'])
                            return msg

                    else:
                        raise ValueError('the job objective type is not supported')

                    log_time_accuracy(job_name,
                                      cur_accuracy,
                                      shared_runtime_history,
                                      job_epoch_dict,
                                      shared_accuracy_history)

                    slot_end_marker = timer()
                    running_slot_time = slot_end_marker - slot_start_marker

                # save the model/job since the job has run for the current slot but doesn't achieve SLO
                saver.save(sess, checkpoint_file)

    # exceed the running slot and haven't achieve goal so put the job back to the queue
    job_queue_others.put(job_data)

    msg = 'job {} is finished the current running slot'.format(job_data['id'])
    return msg


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

    for s in ml_workload:
        print(s)

    #######################################################
    # prepare everything necessary
    #######################################################

    # list with runtime-slo jobs only
    ml_workload_runtime = mp.Manager().list()

    # list with other slo jobs
    ml_workload_others = mp.Manager().list()
    # queue for checking if all jobs are completed
    job_queue_others = mp.Manager().Queue()

    # dict for storing job's current accuracy
    job_accuracy_dict = mp.Manager().dict()
    # dict for storing job's overall running time
    job_runtime_dict = mp.Manager().dict()
    # dict for storing job's overall training epochs
    job_epoch_dict = mp.Manager().dict()
    # dict for storing job's completion time (achieving SLO or being terminated)
    job_completion_time_dict = mp.Manager().dict()
    # dict for storing if the job achieving the SLO
    job_attain_dict = mp.Manager().dict()

    # dict for tracking epochs with wall-time
    job_runtime_history = dict()
    # dict for tracking accuracy with wall-time
    job_accuracy_history = dict()

    proc_pool = mp.Pool(num_gpu, maxtasksperchild=1)

    # init dicts
    for job in ml_workload:
        if job['goal_type'] == 'runtime':
            ml_workload_runtime.append(job)
        else:
            ml_workload_others.append(job)
            job_queue_others.put(job)

        job_key = str(job['id']) + '-' + job['model']

        job_accuracy_dict[job_key] = 0
        job_runtime_dict[job_key] = 0
        job_epoch_dict[job_key] = 0
        job_completion_time_dict[job_key] = 0
        job_attain_dict[job_key] = 0

        init_sub_accuracy_list = mp.Manager().list()
        init_sub_accuracy_list.append('0:' + proc_start_time)
        job_accuracy_history[job_key] = init_sub_accuracy_list
        init_sub_runtime_list = mp.Manager().list()
        init_sub_runtime_list.append('0:' + proc_start_time)
        job_runtime_history[job_key] = init_sub_runtime_list

    while len(ml_workload_runtime) != 0:
        results_runtime = list()
        runtime_job_num = len(ml_workload_runtime)
        for idx in range(runtime_job_num):
            gpuid = idx % num_gpu
            result = proc_pool.apply_async(train_job_runtime, args=(gpuid, job_runtime_history, job_accuracy_history))
            results_runtime.append(result)

        for i in results_runtime:
            i.wait()

        for i in results_runtime:
            if i.ready():
                if i.successful():
                    print(i.get())

    while not job_queue_others.empty():
        results_others = list()
        for idx in range(job_queue_others.qsize()):
            gpuid = idx % num_gpu
            result = proc_pool.apply_async(train_job_others, args=(gpuid, job_runtime_history, job_accuracy_history))
            results_others.append(result)

        for i in results_others:
            i.wait()

        for i in results_others:
            if i.ready():
                if i.successful():
                    print(i.get())

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
