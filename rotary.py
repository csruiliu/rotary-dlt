import tensorflow as tf
import multiprocessing as mp
from timeit import default_timer as timer
import os
import queue
import math
from datetime import datetime

import config.config_rotary as cfg_rotary
import config.config_path as cfg_path
from workload.cv_generator import CVWorkloadGenerator
from estimator.dl_estimator import DLEstimator
from workload.tensorflow_cifar.tools.dataset_loader import load_cifar10_keras
from utils.model_tool import build_model


def log_time_accuracy(job_instance_key,
                      current_acc,
                      shared_time_dict,
                      shared_accuracy_dict):
    now_time_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    sub_runtime_list = shared_time_dict[job_instance_key]
    sub_runtime_list.append(str(job_epoch_dict[job_instance_key]) + ':' + now_time_date)
    shared_time_dict[job_instance_key] = sub_runtime_list

    sub_accuracy_list = shared_accuracy_dict[job_instance_key]
    sub_accuracy_list.append(str(current_acc) + ':' + now_time_date)
    shared_accuracy_dict[job_instance_key] = sub_accuracy_list


def train_job_trial(gpu_id,
                    shared_runtime_history,
                    shared_accuracy_history):
    trial_slot_start_marker = timer()

    try:
        job_data = job_queue_trial.get_nowait()
    except queue.Empty:
        return

    job_name = str(job_data['id']) + '-' + job_data['model']
    print('get job {}'.format(job_name))

    assign_device = '/gpu:' + str(gpu_id)
    print('running job {} on device {} at process {}'.format(job_name, assign_device, os.getpid()))

    model_opt = job_data['opt']
    model_learn_rate = job_data['learn_rate']
    train_batchsize = job_data['batch_size']

    img_w = 32
    img_h = 32
    num_chn = 3
    num_cls = 10

    train_feature, train_labels, eval_feature, eval_labels = load_cifar10_keras()

    with tf.device(assign_device):
        feature_ph = tf.placeholder(tf.float32, [None, img_w, img_h, num_chn])
        label_ph = tf.placeholder(tf.int64, [None, num_cls])
        train_op, eval_op, total_parameters = build_model(job_data,
                                                          model_opt,
                                                          model_learn_rate,
                                                          num_cls,
                                                          feature_ph,
                                                          label_ph)

        # store the total parameters of the model to dict
        job_parameters_dict[job_name] = total_parameters

        # ready to train the job
        saver = tf.train.Saver()

        model_ckpt_save_path = cfg_path.ckpt_save_path + '/' + job_name

        if os.path.exists(model_ckpt_save_path):
            os.makedirs(model_ckpt_save_path)

        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            checkpoint_file = model_ckpt_save_path + '/model_ckpt'
            sess.run(tf.global_variables_initializer())

            num_batch = train_labels.shape[0] // train_batchsize

            epochtime_start_marker = timer()

            for n in range(num_batch):
                # print('step {} / {}'.format(n + 1, num_batch))
                batch_offset = n * train_batchsize
                batch_end = (n + 1) * train_batchsize

                train_data_batch = train_feature[batch_offset:batch_end]
                train_label_batch = train_labels[batch_offset:batch_end]

                sess.run(train_op, feed_dict={feature_ph: train_data_batch, label_ph: train_label_batch})

            epochtime_end_marker = timer()

            # compute the time of single epoch time for the job
            job_epochtime_dict[job_name] = epochtime_end_marker - epochtime_start_marker

            # start evaluation phrase
            print('start evaluating job {} at process {}'.format(job_name, os.getpid()))
            acc_sum = 0
            eval_batch_size = 50
            num_batch_eval = eval_labels.shape[0] // eval_batch_size
            for ne in range(num_batch_eval):
                # print('evaluation step %d / %d' % (ne + 1, num_batch_eval))
                batch_offset = ne * eval_batch_size
                batch_end = (ne + 1) * eval_batch_size
                eval_feature_batch = eval_feature[batch_offset:batch_end]
                eval_label_batch = eval_labels[batch_offset:batch_end]
                acc_batch = sess.run(eval_op,
                                     feed_dict={feature_ph: eval_feature_batch, label_ph: eval_label_batch})
                acc_sum += acc_batch

            cur_accuracy = acc_sum / num_batch_eval
            print('evaluation accuracy:{}'.format(cur_accuracy))

            pre_accuracy = job_accuracy_dict[job_name]

            trial_slot_end_marker = timer()

            job_runtime_dict[job_name] += trial_slot_end_marker - trial_slot_start_marker
            job_epoch_dict[job_name] += 1
            job_accuracy_dict[job_name] = cur_accuracy

            # add this trial result to estimator
            dl_estimator.add_actual_data(job_key=job_name, accuracy=cur_accuracy, epoch=1)

            if job_data['goal_type'] == 'accuracy':
                if job_accuracy_dict[job_name] >= job_data['goal_value']:
                    end_time_overall = timer()
                    job_completion_time_dict[job_name] = end_time_overall - start_time_overall
                    job_attain_dict[job_name] = 1
                    log_time_accuracy(job_name, cur_accuracy, shared_runtime_history, shared_accuracy_history)
                    saver.save(sess, checkpoint_file)
                    msg = 'job {} reaches SLO'.format(job_data['id'])
                    return msg

                if job_epoch_dict[job_name] >= job_data['goal_value_extra']:
                    end_time_overall = timer()
                    job_completion_time_dict[job_name] = end_time_overall - start_time_overall
                    saver.save(sess, checkpoint_file)
                    log_time_accuracy(job_name, cur_accuracy, shared_runtime_history, shared_accuracy_history)
                    msg = 'job {} is finished'.format(job_data['id'])
                    return msg

            elif job_data['goal_type'] == 'convergence':
                delta = cur_accuracy - pre_accuracy
                if delta <= job_data['goal_value']:
                    end_time_overall = timer()
                    job_completion_time_dict[job_name] = end_time_overall - start_time_overall
                    job_attain_dict[job_name] = 1
                    saver.save(sess, checkpoint_file)
                    log_time_accuracy(job_name, cur_accuracy, shared_runtime_history, shared_accuracy_history)
                    msg = 'job {} reaches the SLO'.format(job_data['id'])
                    return msg

                if job_epoch_dict[job_name] >= job_data['goal_value_extra']:
                    end_time_overall = timer()
                    job_completion_time_dict[job_name] = end_time_overall - start_time_overall
                    saver.save(sess, checkpoint_file)
                    log_time_accuracy(job_name, cur_accuracy, shared_runtime_history, shared_accuracy_history)
                    msg = 'job {} is finished'.format(job_data['id'])
                    return msg

            elif job_data['goal_type'] == 'runtime':
                if job_epoch_dict[job_name] >= job_data['goal_value']:
                    end_time_overall = timer()
                    job_completion_time_dict[job_name] = end_time_overall - start_time_overall
                    job_attain_dict[job_name] = 1
                    saver.save(sess, checkpoint_file)
                    msg = 'job {} reaches the SLO'.format(job_data['id'])
                    log_time_accuracy(job_name, cur_accuracy, shared_runtime_history, shared_accuracy_history)
                    return msg
            else:
                raise ValueError('the job objective type is not supported')

            # save the model and exit the trial slot
            saver.save(sess, checkpoint_file)
            log_time_accuracy(job_name, cur_accuracy, shared_runtime_history, shared_accuracy_history)

    msg = 'job {} is finished the current running slot'.format(job_data['id'])
    return msg


def train_job(gpu_id,
              job_data,
              shared_runtime_history,
              shared_accuracy_history):
    preparation_start_marker = timer()
    slot_start_marker = timer()

    # count the training time of this slot
    running_slot_time = 0

    ################################################################
    # randomly pop a job key from the queue_anony
    ################################################################
    try:
        job_anony = job_queue_anony.get_nowait()
    except queue.Empty:
        return

    ################################################################
    # start working on the selected job
    ################################################################

    job_name = str(job_data['id']) + '-' + job_data['model']
    print('get job {}'.format(job_name))

    # get the device id
    assign_device = '/gpu:' + str(gpu_id)
    print('running on device {}'.format(assign_device))

    # get opt, learning rate, batch size and image size for training
    model_opt = job_data['opt']
    model_learn_rate = job_data['learn_rate']
    train_batchsize = job_data['batch_size']

    img_w = 32
    img_h = 32
    num_chn = 3
    num_cls = 10

    # load cifar10 data
    train_feature, train_labels, eval_feature, eval_labels = load_cifar10_keras()

    with tf.device(assign_device):
        feature_ph = tf.placeholder(tf.float32, [None, img_w, img_h, num_chn])
        label_ph = tf.placeholder(tf.int64, [None, num_cls])
        train_op, eval_op, total_parameters = build_model(job_data,
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

        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            # check if the checkpoint file exist
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

                for i in range(num_batch):
                    # print('step {} / {}'.format(i + 1, num_batch))
                    batch_offset = i * train_batchsize
                    batch_end = (i + 1) * train_batchsize

                    train_data_batch = train_feature[batch_offset:batch_end]
                    train_label_batch = train_labels[batch_offset:batch_end]

                    sess.run(train_op, feed_dict={feature_ph: train_data_batch, label_ph: train_label_batch})

                print('start evaluation phrase')
                acc_sum = 0
                eval_batch_size = 50
                num_batch_eval = eval_labels.shape[0] // eval_batch_size
                for i in range(num_batch_eval):
                    # print('evaluation step %d / %d' % (i + 1, num_batch_eval))
                    batch_offset = i * eval_batch_size
                    batch_end = (i + 1) * eval_batch_size
                    eval_feature_batch = eval_feature[batch_offset:batch_end]
                    eval_label_batch = eval_labels[batch_offset:batch_end]
                    acc_batch = sess.run(eval_op,
                                         feed_dict={feature_ph: eval_feature_batch, label_ph: eval_label_batch})
                    acc_sum += acc_batch

                cur_accuracy = acc_sum / num_batch_eval
                print('evaluation accuracy:{}'.format(cur_accuracy))

                pre_accuracy = job_accuracy_dict[job_name]

                epoch_end_marker = timer()

                # tracking the time and accuracy
                job_runtime_dict[job_name] += epoch_end_marker - epoch_start_marker
                job_epoch_dict[job_name] += 1
                job_accuracy_dict[job_name] = cur_accuracy

                # add current results
                dl_estimator.add_actual_data(job_name, cur_accuracy, job_epoch_dict[job_name])

                if job_data['goal_type'] == 'accuracy':
                    if job_accuracy_dict[job_name] >= job_data['goal_value']:
                        end_time_overall = timer()
                        job_completion_time_dict[job_name] = end_time_overall - start_time_overall
                        job_attain_dict[job_name] = 1
                        saver.save(sess, checkpoint_file)
                        log_time_accuracy(job_name, cur_accuracy, shared_runtime_history, shared_accuracy_history)
                        msg = 'job {} reaches the SLO'.format(job_data['id'])
                        return msg

                    if job_epoch_dict[job_name] >= job_data['goal_value_extra']:
                        end_time_overall = timer()
                        job_completion_time_dict[job_name] = end_time_overall - start_time_overall
                        saver.save(sess, checkpoint_file)
                        log_time_accuracy(job_name, cur_accuracy, shared_runtime_history, shared_accuracy_history)
                        msg = 'job {} is finished'.format(job_data['id'])
                        return msg

                elif job_data['goal_type'] == 'convergence':
                    delta = cur_accuracy - pre_accuracy
                    if delta <= job_data['goal_value']:
                        end_time_overall = timer()
                        job_completion_time_dict[job_name] = end_time_overall - start_time_overall
                        job_attain_dict[job_name] = 1
                        saver.save(sess, checkpoint_file)
                        log_time_accuracy(job_name, cur_accuracy, shared_runtime_history, shared_accuracy_history)
                        msg = 'job {} reaches the SLO'.format(job_data['id'])
                        return msg

                    if job_epoch_dict[job_name] >= job_data['goal_value_extra']:
                        end_time_overall = timer()
                        job_completion_time_dict[job_name] = end_time_overall - start_time_overall
                        saver.save(sess, checkpoint_file)
                        log_time_accuracy(job_name, cur_accuracy, shared_runtime_history, shared_accuracy_history)
                        msg = 'job {} is finished'.format(job_data['id'])
                        return msg

                elif job_data['goal_type'] == 'runtime':
                    if job_epoch_dict[job_name] >= job_data['goal_value']:
                        end_time_overall = timer()
                        job_completion_time_dict[job_name] = end_time_overall - start_time_overall
                        job_attain_dict[job_name] = 1
                        saver.save(sess, checkpoint_file)
                        log_time_accuracy(job_name, cur_accuracy, shared_runtime_history, shared_accuracy_history)
                        msg = 'job {} reaches the SLO'.format(job_data['id'])
                        return msg
                else:
                    raise ValueError('the job objective type is not supported')

                log_time_accuracy(job_name, cur_accuracy, shared_runtime_history, shared_accuracy_history)

                slot_end_marker = timer()
                running_slot_time = slot_end_marker - slot_start_marker

    # exceed the running slot and haven't achieve goal so put the job back to the queue
    job_list_rotary.append(job_data)
    job_queue_anony.put(job_anony)

    # save the model/job since the job has run for the current slot but doesn't achieve SLO
    saver.save(sess, checkpoint_file)
    msg = 'job {} is finished the current running slot'.format(job_data['id'])
    return msg


if __name__ == "__main__":
    proc_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print('============= the whole exp starts at: {}===================='.format(proc_start_time))

    # start timing for job completion
    start_time_overall = timer()

    # create folder if not exist
    #if not os.path.exists(cfg_path.ckpt_save_path):
    #    os.makedirs(cfg_path.ckpt_save_path)

    #######################################################
    # get parameters from configuration
    #######################################################

    num_gpu = cfg_rotary.num_gpu
    running_slot = cfg_rotary.running_slot

    assert cfg_rotary.cv_light_ratio + cfg_rotary.cv_med_ratio + cfg_rotary.cv_heavy_ratio == 1

    #######################################################
    # generate the workload
    #######################################################

    wg = CVWorkloadGenerator(cfg_rotary.workload_size,
                             cfg_rotary.cv_light_ratio,
                             cfg_rotary.cv_med_ratio,
                             cfg_rotary.cv_heavy_ratio,
                             cfg_rotary.convergence_ratio,
                             cfg_rotary.accuracy_ratio,
                             cfg_rotary.runtime_ratio,
                             cfg_rotary.random_seed)

    ml_workload = wg.generate_workload()

    for i in ml_workload:
        print(i)

    #######################################################
    # init the accuracy estimator
    #######################################################

    dl_estimator = DLEstimator(topk=5)

    # read all the accuracy file
    for f in os.listdir('./knowledgebase'):
        model_acc_file = os.getcwd() + '/knowledgebase/' + f
        dl_estimator.import_accuracy_dataset(model_acc_file)

    dl_estimator.prepare_workload(ml_workload)

    #######################################################
    # prepare everything necessary
    #######################################################

    # queue for trial phrase
    job_queue_trial = mp.Manager().Queue()

    # queue for checking if all jobs are completed
    job_queue_anony = mp.Manager().Queue()

    # list for rotary phrase
    job_list_rotary = mp.Manager().list()

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

    # dict for tracking epochs with wall-time
    job_runtime_history = dict()
    # dict for tracking accuracy with wall-time
    job_accuracy_history = dict()

    # init process pool
    proc_pool = mp.Pool(num_gpu, maxtasksperchild=1)

    # init some dicts to track the progress
    for job in ml_workload:
        job_key = str(job['id']) + '-' + job['model']
        job_queue_trial.put(job)
        job_list_rotary.append(job)
        job_queue_anony.put(job_key)

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
        gpuid = idx % num_gpu
        result = proc_pool.apply_async(train_job_trial, args=(gpuid,
                                                              job_runtime_history,
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

    #######################################################
    # start the rotary process
    #######################################################

    while not job_queue_anony.empty():
        results_rotary = list()
        job_select = job_list_rotary[0]
        for idx in range(job_queue_anony.qsize()):
            gpuid = idx % num_gpu

            r_score_min = float('inf')

            for job_ins in job_list_rotary:
                job_ins_key = str(job_ins['id']) + '-' + job_ins['model']

                if job_ins['goal_type'] == 'runtime':
                    current_epoch = job_epoch_dict[job_ins_key]
                    r_score = current_epoch / job_ins['goal_value']
                elif job_ins['goal_type'] == 'accuracy':
                    current_epoch = job_epoch_dict[job_ins_key]
                    estimate_all_epoch = dl_estimator.predict_epoch(job_ins, job_ins['goal_value'])
                    if estimate_all_epoch > job_ins['goal_value_extra']:
                        r_score = current_epoch / job_ins['goal_value_extra']
                    else:
                        r_score = current_epoch / estimate_all_epoch
                elif job_ins['goal_type'] == 'convergence':
                    current_epoch = job_epoch_dict[job_ins_key]
                    current_accuracy = job_accuracy_dict[job_ins_key]
                    expected_accuracy = current_accuracy + job_ins['goal_value']
                    estimate_all_epoch = dl_estimator.predict_epoch(job_ins, expected_accuracy)
                    if estimate_all_epoch > job_ins['goal_value_extra']:
                        r_score = current_epoch / job_ins['goal_value_extra']
                    else:
                        r_score = current_epoch / estimate_all_epoch
                else:
                    raise ValueError('the job objective type is not supported')

                if r_score_min > r_score:
                    r_score_min = r_score
                    job_select = job_ins

            try:
                job_list_rotary.remove(job_select)
            except:
                msg = 'job has been handled by other GPU'
                continue

            result = proc_pool.apply_async(train_job, args=(gpuid,
                                                            job_select,
                                                            job_runtime_history,
                                                            job_accuracy_history))
            results_rotary.append(result)

        for i in results_rotary:
            i.wait()

        for i in results_rotary:
            if i.ready():
                if i.successful():
                    print(i.get())

    #######################################################
    # printout the log information
    #######################################################

    for key in job_accuracy_dict:
        print('{} [accuracy]-> {}'.format(key, job_accuracy_dict[key]))

    for key in job_runtime_dict:
        print('{} [accuracy]-> {}'.format(key, job_runtime_dict[key]))

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
