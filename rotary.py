import tensorflow as tf
import multiprocessing as mp
from timeit import default_timer as timer
import os
import queue
import math

import config.config_rotary as cfg_rotary
import config.config_path as cfg_path

from workload.cv_generator import CVWorkloadGenerator
from estimator.accuracy_estimator import AccuracyEstimator
from workload.tensorflow_cifar.tools.dataset_loader import load_cifar10_keras
from utils.model_tool import build_model


def train_job_trial(gpu_id):
    start_time_proc = timer()

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
        train_op, eval_op, model_name, total_parameters = build_model(job_data,
                                                                      model_opt,
                                                                      model_learn_rate,
                                                                      num_cls,
                                                                      feature_ph,
                                                                      label_ph)

        # store the total parameters of the model to dict
        job_parameters_dict[job_name] = total_parameters

        # ready to train the job
        saver = tf.train.Saver()

        model_ckpt_save_path = cfg_path.ckpt_save_path + '/job_' + model_name

        if os.path.exists(model_ckpt_save_path):
            os.makedirs(model_ckpt_save_path)

        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.gpu_options.allow_growth = True



        with tf.Session(config=config) as sess:
            checkpoint_file = model_ckpt_save_path + '/model_ckpt'
            sess.run(tf.global_variables_initializer())

            num_batch = train_labels.shape[0] // train_batchsize

            start_epochtime = timer()
            for n in range(num_batch):
                # print('step {} / {}'.format(n + 1, num_batch))
                batch_offset = n * train_batchsize
                batch_end = (n + 1) * train_batchsize

                train_data_batch = train_feature[batch_offset:batch_end]
                train_label_batch = train_labels[batch_offset:batch_end]

                sess.run(train_op, feed_dict={feature_ph: train_data_batch, label_ph: train_label_batch})

            end_epochtime = timer()

            epochtime = (end_epochtime - start_epochtime)
            job_epochtime_dict[job_name] = epochtime

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

            end_time_proc = timer()
            run_time_proc = end_time_proc - start_time_proc

            run_epoch = 1

            # add this trial result to estimator
            acc_estimator.add_actual_accuracy(job_key=job_name, accuracy=cur_accuracy, epoch=run_epoch)

            pre_accuracy = job_accuracy_dict[job_name]

            job_time_dict[job_name] += run_time_proc
            job_epoch_dict[job_name] += run_epoch
            job_accuracy_dict[job_name] = cur_accuracy

            if job_data['goal_type'] == 'accuracy':
                if (job_accuracy_dict[job_name] > job_data['goal_value'] or
                        job_epoch_dict[job_name] > job_data['goal_value_extra']):
                    saver.save(sess, checkpoint_file)
                    msg = 'job {} is finished'.format(job_data['id'])
                    return msg

            elif job_data['goal_type'] == 'deadline':
                if job_time_dict[job_name] > job_data['goal_value']:
                    saver.save(sess, checkpoint_file)
                    msg = 'job {} is finished'.format(job_data['id'])
                    return msg

            elif job_data['goal_type'] == 'convergence':
                delta = cur_accuracy - pre_accuracy
                if (delta < job_data['goal_value'] or
                        job_epoch_dict[job_name] > job_data['goal_value_extra']):
                    saver.save(sess, checkpoint_file)
                    msg = 'job {} is finished'.format(job_data['id'])
                    return msg

            elif job_data['goal_type'] == 'runtime':
                if job_epoch_dict[job_name] > job_data['goal_value']:
                    saver.save(sess, checkpoint_file)
                    msg = 'job {} is finished'.format(job_data['id'])
                    return msg
            else:
                raise ValueError('the job objective type is not supported')

            saver.save(sess, checkpoint_file)

    msg = 'job {} is finished the current running slot'.format(job_data['id'])
    return msg


def train_job(gpu_id, job_data):
    # start time counting for the whole process
    start_time_proc = timer()
    run_time_proc = 0
    run_epoch = 0

    ################################################################
    # randomly pop a job key from the queue_anony
    ################################################################

    try:
        job_anony = job_queue_anony.get_nowait()
    except queue.Empty:
        return
    '''
    ################################################################
    # get the job according to the cost model
    ################################################################
    epsilon = 3
    r_score_max = 0
    job_data = None

    for job_ins in job_list_rotary:
        job_ins_key = str(job_ins['id']) + '-' + job_ins['model']

        prev_accuracy = job_accuracy_dict[job_ins_key]

        running_epoch = running_slot / job_epochtime_dict[job_ins_key]

        next_accuracy_predict = acc_estimator.predict_accuracy(job_ins, running_epoch)

        delta_accuracy = next_accuracy_predict - prev_accuracy if next_accuracy_predict > prev_accuracy else 0

        r_score = delta_accuracy / running_slot

        if job_ins['goal_type'] == 'deadline':
            if deadline_max == deadline_min:
                r_score += math.exp(epsilon)
            else:
                r_score += math.exp(epsilon * (job_ins['goal_value'] - deadline_min) / (deadline_max - deadline_min))

        if r_score > r_score_max:
            r_score_max = r_score
            job_data = job_ins

    try:
        job_list_rotary.remove(job_data)
    except:
        msg = 'job has been handled by other GPU'
        return msg
    '''
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
        train_op, eval_op, model_name, _ = build_model(job_data,
                                                       model_opt,
                                                       model_learn_rate,
                                                       num_cls,
                                                       feature_ph,
                                                       label_ph)

        # init the tf saver for checkpoint
        saver = tf.train.Saver()

        # get the path of checkpoint
        model_ckpt_save_path = cfg_path.ckpt_save_path + '/job_' + model_name

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

            # check if the total runtime is less than running_slot
            while run_time_proc < running_slot:
                for i in range(num_batch):
                    print('step {} / {}'.format(i + 1, num_batch))
                    batch_offset = i * train_batchsize
                    batch_end = (i + 1) * train_batchsize

                    train_data_batch = train_feature[batch_offset:batch_end]
                    train_label_batch = train_labels[batch_offset:batch_end]

                    sess.run(train_op, feed_dict={feature_ph: train_data_batch, label_ph: train_label_batch})

                run_epoch += 1

                print('start evaluation phrase')
                acc_sum = 0
                eval_batch_size = 50
                num_batch_eval = eval_labels.shape[0] // eval_batch_size
                for i in range(num_batch_eval):
                    print('evaluation step %d / %d' % (i + 1, num_batch_eval))
                    batch_offset = i * eval_batch_size
                    batch_end = (i + 1) * eval_batch_size
                    eval_feature_batch = eval_feature[batch_offset:batch_end]
                    eval_label_batch = eval_labels[batch_offset:batch_end]
                    acc_batch = sess.run(eval_op,
                                         feed_dict={feature_ph: eval_feature_batch, label_ph: eval_label_batch})
                    acc_sum += acc_batch

                cur_accuracy = acc_sum / num_batch_eval
                print('evaluation accuracy:{}'.format(cur_accuracy))

                end_time_proc = timer()
                run_time_proc = end_time_proc - start_time_proc

                acc_estimator.add_actual_accuracy(job_name, cur_accuracy, run_epoch)

                # add this trial result to estimator
                pre_accuracy = job_accuracy_dict[job_name]

                job_time_dict[job_name] += run_time_proc
                job_epoch_dict[job_name] += run_epoch
                job_accuracy_dict[job_name] = cur_accuracy

                if job_data['goal_type'] == 'accuracy':
                    if (job_accuracy_dict[job_name] > job_data['goal_value'] or
                            job_epoch_dict[job_name] >= job_data['goal_value_extra']):
                        saver.save(sess, checkpoint_file)
                        msg = 'job {} is finished'.format(job_data['id'])
                        return msg

                elif job_data['goal_type'] == 'deadline':
                    if round(job_time_dict[job_name]) >= job_data['goal_value']:
                        saver.save(sess, checkpoint_file)
                        msg = 'job {} is finished'.format(job_data['id'])
                        return msg

                elif job_data['goal_type'] == 'convergence':
                    delta = cur_accuracy - pre_accuracy
                    if (delta < job_data['goal_value'] or
                            job_epoch_dict[job_name] >= job_data['goal_value_extra']):
                        saver.save(sess, checkpoint_file)
                        msg = 'job {} is finished'.format(job_data['id'])
                        return msg

                elif job_data['goal_type'] == 'runtime':
                    if job_epoch_dict[job_name] >= job_data['goal_value']:
                        saver.save(sess, checkpoint_file)
                        msg = 'job {} is finished'.format(job_data['id'])
                        return msg
                else:
                    raise ValueError('the job objective type is not supported')

                saver.save(sess, checkpoint_file)

    # exceed the running slot and haven't achieve goal so put the job back to the queue
    job_list_rotary.append(job_data)
    job_queue_anony.put(job_anony)

    msg = 'job {} is finished the current running slot'.format(job_data['id'])
    return msg


if __name__ == "__main__":
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

    wg = CVWorkloadGenerator(cfg_rotary.workload_size,
                             cfg_rotary.cv_light_ratio,
                             cfg_rotary.cv_med_ratio,
                             cfg_rotary.cv_heavy_ratio,
                             cfg_rotary.convergence_ratio,
                             cfg_rotary.accuracy_ratio,
                             cfg_rotary.runtime_ratio,
                             cfg_rotary.deadline_ratio,
                             cfg_rotary.random_seed)

    ml_workload = wg.generate_workload()

    for i in ml_workload:
        print(i)

    #######################################################
    # init the accuracy estimator
    #######################################################

    acc_estimator = AccuracyEstimator(topk=5)

    # read all the accuracy file
    for f in os.listdir('./knowledgebase'):
        model_acc_file = os.getcwd() + '/knowledgebase/' + f
        acc_estimator.import_accuracy_dataset(model_acc_file)

    acc_estimator.import_workload(ml_workload)

    #######################################################
    # data structures for tracking progress of all jobs
    #######################################################

    job_queue_trial = mp.Manager().Queue()
    job_queue_anony = mp.Manager().Queue()
    job_list_rotary = mp.Manager().list()

    job_accuracy_dict = mp.Manager().dict()
    job_time_dict = mp.Manager().dict()
    job_epoch_dict = mp.Manager().dict()
    job_epochtime_dict = mp.Manager().dict()
    job_parameters_dict = mp.Manager().dict()

    deadline_max = 0
    deadline_min = float('inf')

    # init some dicts to track the progress
    for job in ml_workload:
        job_key = str(job['id']) + '-' + job['model']
        job_queue_trial.put(job)
        job_list_rotary.append(job)
        job_queue_anony.put(job_key)

        job_accuracy_dict[job_key] = 0
        job_time_dict[job_key] = 0
        job_epoch_dict[job_key] = 0
        job_epochtime_dict[job_key] = 0
        job_parameters_dict[job_key] = 0

        if job['goal_type'] == 'deadline':
            if job['goal_value'] < deadline_min:
                deadline_min = job['goal_value']
            if job['goal_value'] > deadline_max:
                deadline_max = job['goal_value']

    proc_pool = mp.Pool(num_gpu, maxtasksperchild=1)

    #######################################################
    # start the trial process
    #######################################################

    results_trial = list()
    for idx in range(len(ml_workload)):
        gpuid = idx % num_gpu
        result = proc_pool.apply_async(train_job_trial, args=(gpuid,))
        results_trial.append(result)

    for i in results_trial:
        i.wait()

    for i in results_trial:
        if i.ready():
            if i.successful():
                print(i.get())

    for key in job_accuracy_dict:
        print(key, '[accuracy]->', job_accuracy_dict[key])

    for key in job_time_dict:
        print(key, '[time]->', job_time_dict[key])

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

            epsilon = 3
            r_score_max = float('-inf')

            for job_ins in job_list_rotary:
                job_ins_key = str(job_ins['id']) + '-' + job_ins['model']

                prev_accuracy = job_accuracy_dict[job_ins_key]

                running_epoch = running_slot / job_epochtime_dict[job_ins_key]

                next_accuracy_predict = acc_estimator.predict_accuracy(job_ins, running_epoch)

                delta_accuracy = next_accuracy_predict - prev_accuracy if next_accuracy_predict > prev_accuracy else 0

                r_score = delta_accuracy / running_slot

                if job_ins['goal_type'] == 'deadline':
                    if deadline_max == deadline_min:
                        r_score += math.exp(epsilon)
                    else:
                        r_score += math.exp(
                            epsilon * (job_ins['goal_value'] - deadline_min) / (deadline_max - deadline_min))

                if r_score > r_score_max:
                    r_score_max = r_score
                    job_select = job_ins

            try:
                job_list_rotary.remove(job_select)
            except:
                msg = 'job has been handled by other GPU'
                continue

            result = proc_pool.apply_async(train_job, args=(gpuid, job_select,))
            results_rotary.append(result)

        for i in results_rotary:
            i.wait()

        for i in results_rotary:
            if i.ready():
                if i.successful():
                    print(i.get())

    for key in job_accuracy_dict:
        print(key, '[accuracy]->', job_accuracy_dict[key])

    for key in job_time_dict:
        print(key, '[time]->', job_time_dict[key])

    for key in job_epoch_dict:
        print(key, '[epoch]->', job_epoch_dict[key])
