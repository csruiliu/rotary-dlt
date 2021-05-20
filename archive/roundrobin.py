import tensorflow as tf
import multiprocessing as mp
from timeit import default_timer as timer
import os
import queue
import datetime

import config.config_rotary as cfg_rotary
from workload.cv_generator import CVWorkloadGenerator
from workload.tensorflow_cifar.tools.dataset_loader import load_cifar10_keras
import config.config_path as cfg_path
from utils.model_tool import build_model


def train_job(gpu_id):
    start_time_proc = timer()
    run_time_proc = 0
    run_epoch = 0

    # get the job id
    try:
        job_data = job_queue.get_nowait()
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

                pre_accuracy = job_accuracy_dict[job_name]

                job_time_dict[job_name] += run_time_proc
                job_epoch_dict[job_name] += run_epoch
                job_accuracy_dict[job_name] = cur_accuracy

                if job_data['goal_type'] == 'accuracy':
                    if job_accuracy_dict[job_name] > job_data['goal_value']:
                        end_time_overall = timer()
                        job_completion_time_dict[job_name] = end_time_overall - start_time_overall
                        job_attain_dict[job_name] = 1
                        saver.save(sess, checkpoint_file)
                        msg = 'job {} is finished'.format(job_data['id'])
                        return msg

                    if job_epoch_dict[job_name] >= job_data['goal_value_extra']:
                        end_time_overall = timer()
                        job_completion_time_dict[job_name] = end_time_overall - start_time_overall
                        saver.save(sess, checkpoint_file)
                        msg = 'job {} is finished'.format(job_data['id'])
                        return msg

                elif job_data['goal_type'] == 'deadline':
                    if round(job_time_dict[job_name]) >= job_data['goal_value']:
                        end_time_overall = timer()
                        job_completion_time_dict[job_name] = end_time_overall - start_time_overall
                        job_attain_dict[job_name] = 1
                        saver.save(sess, checkpoint_file)
                        msg = 'job {} is finished'.format(job_data['id'])
                        return msg

                elif job_data['goal_type'] == 'convergence':
                    delta = cur_accuracy - pre_accuracy
                    if delta < job_data['goal_value']:
                        end_time_overall = timer()
                        job_completion_time_dict[job_name] = end_time_overall - start_time_overall
                        job_attain_dict[job_name] = 1
                        saver.save(sess, checkpoint_file)
                        msg = 'job {} is finished'.format(job_data['id'])
                        return msg

                    if job_epoch_dict[job_name] >= job_data['goal_value_extra']:
                        end_time_overall = timer()
                        job_completion_time_dict[job_name] = end_time_overall - start_time_overall
                        saver.save(sess, checkpoint_file)
                        msg = 'job {} is finished'.format(job_data['id'])
                        return msg

                elif job_data['goal_type'] == 'runtime':
                    if job_epoch_dict[job_name] >= job_data['goal_value']:
                        end_time_overall = timer()
                        job_completion_time_dict[job_name] = end_time_overall - start_time_overall
                        job_attain_dict[job_name] = 1
                        saver.save(sess, checkpoint_file)
                        msg = 'job {} is finished'.format(job_data['id'])
                        return msg
                else:
                    raise ValueError('the job objective type is not supported')

                saver.save(sess, checkpoint_file)

    # exceed the running slot and haven't achieve goal so put the job back to the queue
    job_queue.put(job_data)

    msg = 'job {} is finished for current running slot'.format(job_data['id'])
    return msg


if __name__ == "__main__":
    if not os.path.exists(cfg_path.ckpt_save_path):
        os.makedirs(cfg_path.ckpt_save_path)

    num_gpu = cfg_rotary.num_gpu
    running_slot = cfg_rotary.running_slot

    assert cfg_rotary.cv_light_ratio + cfg_rotary.cv_med_ratio + cfg_rotary.cv_heavy_ratio == 1

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

    for s in ml_workload:
        print(s)

    now = datetime.datetime.now()

    start_time_overall = timer()

    job_queue = mp.Manager().Queue()

    job_accuracy_dict = mp.Manager().dict()
    job_time_dict = mp.Manager().dict()
    job_epoch_dict = mp.Manager().dict()
    job_completion_time_dict = mp.Manager().dict()
    job_attain_dict = mp.Manager().dict()

    proc_pool = mp.Pool(num_gpu, maxtasksperchild=1)

    # init some dicts to track the progress
    for job in ml_workload:
        job_key = str(job['id']) + '-' + job['model']
        job_accuracy_dict[job_key] = 0
        job_time_dict[job_key] = 0
        job_epoch_dict[job_key] = 0
        job_attain_dict[job_key] = 0
        job_completion_time_dict[job_key] = 0
        job_queue.put(job)

    while not job_queue.empty():
        results = list()
        for idx in range(job_queue.qsize()):
            gpuid = idx % num_gpu
            result = proc_pool.apply_async(train_job, args=(gpuid,))
            results.append(result)

        for i in results:
            i.wait()

        for i in results:
            if i.ready():
                if i.successful():
                    print(i.get())

    print("Start date and time : ")
    print(now.strftime("%Y-%m-%d %H:%M:%S"))

    for key in job_accuracy_dict:
        print(key, '[accuracy]->', job_accuracy_dict[key])

    for key in job_time_dict:
        print(key, '[time]->', job_time_dict[key])

    for key in job_epoch_dict:
        print(key, '[epoch]->', job_epoch_dict[key])

    for key in job_completion_time_dict:
        print(key, '[completion time]->', job_completion_time_dict[key])

    for key in job_attain_dict:
        print(key, '[attainment]->', job_attain_dict[key])