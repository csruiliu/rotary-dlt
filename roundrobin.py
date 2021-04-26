import tensorflow as tf
from multiprocessing import Manager, Pool
from timeit import default_timer as timer
import os

import workload.tensorflow_ptb.tools.ptb_reader as ptb_reader
from workload.tensorflow_cifar.tools.dataset_loader import load_cifar10_keras
import config.config_path as cfg_path
import config.config_workload as cfg_workload
from workload.generator import WorkloadGenerator
from utils.model_tool import build_model

'''
def train_job(job_dict,
              num_gpu,
              time_slot,
              job_accuracy_dict,
              job_time_dict,
              job_done_dict):

    assign_device = '/gpu:' + str(job_data['id'] % num_gpu)

    train_dataset = job_data['dataset']

    if train_dataset == 'cifar10':
        start_time = timer()
        img_w = 32
        img_h = 32
        num_chn = 3
        num_cls = 10

        train_feature, train_labels, eval_feature, eval_labels = load_cifar10_keras()
        ckpt_save_path = cfg_path.ckpt_save_path + '/job_' + str(job_data['id'])

        with tf.device(assign_device):
            feature_ph = tf.placeholder(tf.float32, [None, img_w, img_h, num_chn])
            label_ph = tf.placeholder(tf.int64, [None, num_cls])
            train_op, eval_op, model_name = build_model(job_data, feature_ph, label_ph)

            saver = tf.train.Saver()
            model_ckpt_save_path = ckpt_save_path + '/' + model_name
            if not os.path.exists(model_ckpt_save_path):
                os.makedirs(model_ckpt_save_path)
            checkpoint_file = model_ckpt_save_path + '/' + 'model_ckpt'
            train_batchsize = job_data['batch_size']

            config = tf.ConfigProto()
            config.allow_soft_placement = True
            config.gpu_options.allow_growth = True

            with tf.Session(config=config) as sess:
                if os.path.isfile(checkpoint_file + '.meta'):
                    saver.restore(sess, checkpoint_file)
                else:
                    sess.run(tf.global_variables_initializer())

                num_batch = train_labels.shape[0] // train_batchsize

                while True:
                    for i in range(num_batch):
                        print('step {} / {}'.format(i + 1, num_batch))
                        batch_offset = i * train_batchsize
                        batch_end = (i + 1) * train_batchsize

                        train_data_batch = train_feature[batch_offset:batch_end]
                        train_label_batch = train_labels[batch_offset:batch_end]

                        sess.run(train_op, feed_dict={feature_ph: train_data_batch, label_ph: train_label_batch})

                    cur_accuracy = sess.run(eval_op, feed_dict={feature_ph: eval_feature, label_ph: eval_labels})

                    end_time = timer()
                    dur_time = end_time - start_time

                    pre_accuracy = job_accuracy_dict[model_name]

                    job_time_dict[model_name] += dur_time
                    job_accuracy_dict[model_name] = cur_accuracy

                    if job_data['goal_type'] == 'accuracy':
                        if cur_accuracy >= job_data['goal_value']:
                            job_done_dict[model_name] = True
                            break
                    elif job_data['goal_type'] == 'runtime':
                        if job_time_dict[model_name] > job_data['goal_value']:
                            job_done_dict[model_name] = True
                            break
                    elif job_data['goal_type'] == 'convergence':
                        delta = cur_accuracy - pre_accuracy
                        if delta < job_data['goal_value']:
                            job_done_dict[model_name] = True
                            break
                    else:
                        raise ValueError('the job objective type is not supported')

    else:
        if job_data['model'] in ['word2vec']:
            start_time = timer()
            batch_size = job_data['batch_size']
            ptb_train_data, ptb_valid_data, _, ptb_vocab_size = ptb_reader.ptb_data_skipgram(cfg_workload.ptb_path)

            train_data = list(ptb_reader.ptb_batch_iterator_skipgrams(ptb_train_data, batch_size, ptb_vocab_size))
            eval_data = list(ptb_reader.ptb_batch_iterator_skipgrams(ptb_valid_data, batch_size, ptb_vocab_size))

            num_train_epochs = len(train_data)
            num_eval_epochs = len(eval_data)

            with tf.device(assign_device):
                feature_ph = tf.placeholder(tf.float32, shape=[None, ptb_vocab_size])
                label_ph = tf.placeholder(tf.float32, shape=[None, ptb_vocab_size])

                train_op, eval_op, model_name = build_model(job_data, feature_ph, label_ph)

                saver = tf.train.Saver()
                ckpt_save_path = cfg_path.ckpt_save_path + '/job_' + str(job_data['id'])
                model_ckpt_save_path = ckpt_save_path + '/' + model_name
                if not os.path.exists(model_ckpt_save_path):
                    os.makedirs(model_ckpt_save_path)
                checkpoint_file = model_ckpt_save_path + '/' + 'model_ckpt'

                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                config.allow_soft_placement = True

                with tf.Session(config=config) as sess:
                    if os.path.isfile(checkpoint_file + '.meta'):
                        saver.restore(sess, checkpoint_file)
                    else:
                        sess.run(tf.global_variables_initializer())

                    while True:
                        for e in range(num_train_epochs):
                            print("step {} / {}".format(e + 1, num_train_epochs))
                            train_batch = train_data[e]
                            train_input_batch = train_batch[0]
                            train_target_batch = train_batch[1]
                            sess.run(train_op, feed_dict={feature_ph: train_input_batch,
                                                          label_ph: train_target_batch})

                        sum_accuracy = 0
                        for e in range(num_eval_epochs):
                            print("evaluation eval {} / {}".format(e + 1, num_eval_epochs))
                            eval_batch = eval_data[e]
                            eval_input_batch = eval_batch[0]
                            eval_target_batch = eval_batch[1]
                            eval_batch_accuracy = sess.run(eval_op, feed_dict={feature_ph: eval_input_batch,
                                                                               label_ph: eval_target_batch})
                            sum_accuracy += eval_batch_accuracy

                        cur_accuracy = sum_accuracy / num_eval_epochs

                        end_time = timer()
                        dur_time = end_time - start_time

                        pre_accuracy = job_accuracy_dict[model_name]

                        job_time_dict[model_name] += dur_time
                        job_accuracy_dict[model_name] = cur_accuracy

                        if job_data['goal_type'] == 'accuracy':
                            if cur_accuracy >= job_data['goal_value']:
                                job_done_dict[model_name] = True
                                break
                        elif job_data['goal_type'] == 'runtime':
                            if job_time_dict[model_name] > job_data['goal_value']:
                                job_done_dict[model_name] = True
                                break
                        elif job_data['goal_type'] == 'convergence':
                            delta = cur_accuracy - pre_accuracy
                            if delta < job_data['goal_value']:
                                job_done_dict[model_name] = True
                                break
                        else:
                            raise ValueError('the job objective type is not supported')

        else:
            start_time = timer()
            num_step = 35
            batch_size = job_data['batch_size']

            ptb_train_data, ptb_valid_data, _, ptb_vocab_size = ptb_reader.ptb_data_raw(cfg_workload.ptb_path)

            train_data = list(ptb_reader.ptb_batch_iterator(ptb_train_data, batch_size, num_step, ptb_vocab_size))
            eval_data = list(ptb_reader.ptb_batch_iterator(ptb_valid_data, batch_size, num_step, ptb_vocab_size))

            num_train_epochs = len(train_data)
            num_eval_epochs = len(eval_data)

            with tf.device(assign_device):
                feature_ph = tf.placeholder(tf.float32, [None, num_step, ptb_vocab_size])
                label_ph = tf.placeholder(tf.float32, [None, ptb_vocab_size])

                train_op, eval_op, model_name = build_model(job_data, feature_ph, label_ph)

                saver = tf.train.Saver()
                ckpt_save_path = cfg_path.ckpt_save_path + '/job_' + str(job_data['id'])
                model_ckpt_save_path = ckpt_save_path + '/' + model_name
                if not os.path.exists(model_ckpt_save_path):
                    os.makedirs(model_ckpt_save_path)
                checkpoint_file = model_ckpt_save_path + '/' + 'model_ckpt'

                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                config.allow_soft_placement = True

                with tf.Session(config=config) as sess:
                    if os.path.isfile(checkpoint_file + '.meta'):
                        saver.restore(sess, checkpoint_file)
                    else:
                        sess.run(tf.global_variables_initializer())

                    while True:
                        for e in range(num_train_epochs):
                            print("step {} / {}".format(e + 1, num_train_epochs))
                            train_batch = train_data[e]
                            train_input_batch = train_batch[0]
                            train_target_batch = train_batch[1]
                            sess.run(train_op, feed_dict={feature_ph: train_input_batch,
                                                          label_ph: train_target_batch})

                        sum_accuracy = 0
                        for e in range(num_eval_epochs):
                            print("evaluation eval {} / {}".format(e + 1, num_eval_epochs))
                            eval_batch = eval_data[e]
                            eval_input_batch = eval_batch[0]
                            eval_target_batch = eval_batch[1]
                            eval_batch_accuracy = sess.run(eval_op, feed_dict={feature_ph: eval_input_batch,
                                                                               label_ph: eval_target_batch})
                            sum_accuracy += eval_batch_accuracy

                        cur_accuracy = sum_accuracy / num_eval_epochs

                        end_time = timer()
                        dur_time = end_time - start_time

                        pre_accuracy = job_accuracy_dict[model_name]

                        job_time_dict[model_name] += dur_time
                        job_accuracy_dict[model_name] = cur_accuracy

                        if job_data['goal_type'] == 'accuracy':
                            if cur_accuracy >= job_data['goal_value']:
                                job_done_dict[model_name] = True
                                break
                        elif job_data['goal_type'] == 'runtime':
                            if job_time_dict[model_name] > job_data['goal_value']:
                                job_done_dict[model_name] = True
                                break
                        elif job_data['goal_type'] == 'convergence':
                            delta = cur_accuracy - pre_accuracy
                            if delta < job_data['goal_value']:
                                job_done_dict[model_name] = True
                                break
                        else:
                            raise ValueError('the job objective type is not supported')
'''


def train_job(num_gpu, t_window):
    # job_name = job_queue.get()
    job_data = job_queue.get_nowait()
    job_name = str(job_data['id']) + '-' + job_data['model']
    assign_device = '/gpu:' + str(job_data['id'] % num_gpu)
    train_dataset = job_data['dataset']

    if train_dataset == 'cifar10':
        start_time = timer()
        img_w = 32
        img_h = 32
        num_chn = 3
        num_cls = 10

        train_feature, train_labels, eval_feature, eval_labels = load_cifar10_keras()
        ckpt_save_path = cfg_path.ckpt_save_path + '/job_' + str(job_data['id'])

        with tf.device(assign_device):
            feature_ph = tf.placeholder(tf.float32, [None, img_w, img_h, num_chn])
            label_ph = tf.placeholder(tf.int64, [None, num_cls])
            train_op, eval_op, model_name = build_model(job_data, feature_ph, label_ph)

            saver = tf.train.Saver()
            model_ckpt_save_path = ckpt_save_path + '/' + model_name
            if not os.path.exists(model_ckpt_save_path):
                os.makedirs(model_ckpt_save_path)
            checkpoint_file = model_ckpt_save_path + '/' + 'model_ckpt'
            train_batchsize = job_data['batch_size']

            config = tf.ConfigProto()
            config.allow_soft_placement = True
            config.gpu_options.allow_growth = True

            with tf.Session(config=config) as sess:
                if os.path.isfile(checkpoint_file + '.meta'):
                    saver.restore(sess, checkpoint_file)
                else:
                    sess.run(tf.global_variables_initializer())

                num_batch = train_labels.shape[0] // train_batchsize

                end_time = timer()
                dur_time = end_time - start_time
                while dur_time < t_window:
                    for i in range(num_batch):
                        print('step {} / {}'.format(i + 1, num_batch))
                        batch_offset = i * train_batchsize
                        batch_end = (i + 1) * train_batchsize

                        train_data_batch = train_feature[batch_offset:batch_end]
                        train_label_batch = train_labels[batch_offset:batch_end]

                        sess.run(train_op, feed_dict={feature_ph: train_data_batch, label_ph: train_label_batch})

                    cur_accuracy = sess.run(eval_op, feed_dict={feature_ph: eval_feature, label_ph: eval_labels})

                    end_time = timer()
                    dur_time = end_time - start_time

                    pre_accuracy = job_accuracy_dict[model_name]

                    job_time_dict[model_name] += dur_time
                    job_accuracy_dict[model_name] = cur_accuracy

                    if job_data['goal_type'] == 'accuracy':
                        if cur_accuracy >= job_data['goal_value']:
                            job_done_dict[model_name] = True
                            break
                    elif job_data['goal_type'] == 'runtime':
                        if job_time_dict[model_name] > job_data['goal_value']:
                            job_done_dict[model_name] = True
                            break
                    elif job_data['goal_type'] == 'convergence':
                        delta = cur_accuracy - pre_accuracy
                        if delta < job_data['goal_value']:
                            job_done_dict[model_name] = True
                            break
                    else:
                        raise ValueError('the job objective type is not supported')

    else:
        if job_data['model'] in ['word2vec']:
            start_time = timer()
            batch_size = job_data['batch_size']
            ptb_train_data, ptb_valid_data, _, ptb_vocab_size = ptb_reader.ptb_data_skipgram(cfg_workload.ptb_path)

            train_data = list(ptb_reader.ptb_batch_iterator_skipgrams(ptb_train_data, batch_size, ptb_vocab_size))
            eval_data = list(ptb_reader.ptb_batch_iterator_skipgrams(ptb_valid_data, batch_size, ptb_vocab_size))

            num_train_epochs = len(train_data)
            num_eval_epochs = len(eval_data)

            with tf.device(assign_device):
                feature_ph = tf.placeholder(tf.float32, shape=[None, ptb_vocab_size])
                label_ph = tf.placeholder(tf.float32, shape=[None, ptb_vocab_size])

                train_op, eval_op, model_name = build_model(job_data, feature_ph, label_ph)

                saver = tf.train.Saver()
                ckpt_save_path = cfg_path.ckpt_save_path + '/job_' + str(job_data['id'])
                model_ckpt_save_path = ckpt_save_path + '/' + model_name
                if not os.path.exists(model_ckpt_save_path):
                    os.makedirs(model_ckpt_save_path)
                checkpoint_file = model_ckpt_save_path + '/' + 'model_ckpt'

                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                config.allow_soft_placement = True

                with tf.Session(config=config) as sess:
                    if os.path.isfile(checkpoint_file + '.meta'):
                        saver.restore(sess, checkpoint_file)
                    else:
                        sess.run(tf.global_variables_initializer())

                    end_time = timer()
                    dur_time = end_time - start_time
                    while dur_time < t_window:
                        for e in range(num_train_epochs):
                            print("step {} / {}".format(e + 1, num_train_epochs))
                            train_batch = train_data[e]
                            train_input_batch = train_batch[0]
                            train_target_batch = train_batch[1]
                            sess.run(train_op, feed_dict={feature_ph: train_input_batch,
                                                          label_ph: train_target_batch})

                        sum_accuracy = 0
                        for e in range(num_eval_epochs):
                            print("evaluation eval {} / {}".format(e + 1, num_eval_epochs))
                            eval_batch = eval_data[e]
                            eval_input_batch = eval_batch[0]
                            eval_target_batch = eval_batch[1]
                            eval_batch_accuracy = sess.run(eval_op, feed_dict={feature_ph: eval_input_batch,
                                                                               label_ph: eval_target_batch})
                            sum_accuracy += eval_batch_accuracy

                        cur_accuracy = sum_accuracy / num_eval_epochs

                        end_time = timer()
                        dur_time = end_time - start_time

                        pre_accuracy = job_accuracy_dict[model_name]

                        job_time_dict[model_name] += dur_time
                        job_accuracy_dict[model_name] = cur_accuracy

                        if job_data['goal_type'] == 'accuracy':
                            if cur_accuracy >= job_data['goal_value']:
                                job_done_dict[model_name] = True
                                break
                        elif job_data['goal_type'] == 'runtime':
                            if job_time_dict[model_name] > job_data['goal_value']:
                                job_done_dict[model_name] = True
                                break
                        elif job_data['goal_type'] == 'convergence':
                            delta = cur_accuracy - pre_accuracy
                            if delta < job_data['goal_value']:
                                job_done_dict[model_name] = True
                                break
                        else:
                            raise ValueError('the job objective type is not supported')
        else:
            start_time = timer()
            num_step = 35
            batch_size = job_data['batch_size']

            ptb_train_data, ptb_valid_data, _, ptb_vocab_size = ptb_reader.ptb_data_raw(cfg_workload.ptb_path)

            train_data = list(ptb_reader.ptb_batch_iterator(ptb_train_data, batch_size, num_step, ptb_vocab_size))
            eval_data = list(ptb_reader.ptb_batch_iterator(ptb_valid_data, batch_size, num_step, ptb_vocab_size))

            num_train_epochs = len(train_data)
            num_eval_epochs = len(eval_data)

            with tf.device(assign_device):
                feature_ph = tf.placeholder(tf.float32, [None, num_step, ptb_vocab_size])
                label_ph = tf.placeholder(tf.float32, [None, ptb_vocab_size])

                train_op, eval_op, model_name = build_model(job_data, feature_ph, label_ph)

                saver = tf.train.Saver()
                ckpt_save_path = cfg_path.ckpt_save_path + '/job_' + str(job_data['id'])
                model_ckpt_save_path = ckpt_save_path + '/' + model_name
                if not os.path.exists(model_ckpt_save_path):
                    os.makedirs(model_ckpt_save_path)
                checkpoint_file = model_ckpt_save_path + '/' + 'model_ckpt'

                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                config.allow_soft_placement = True

                with tf.Session(config=config) as sess:
                    if os.path.isfile(checkpoint_file + '.meta'):
                        saver.restore(sess, checkpoint_file)
                    else:
                        sess.run(tf.global_variables_initializer())

                    end_time = timer()
                    dur_time = end_time - start_time
                    while dur_time < t_window:
                        for e in range(num_train_epochs):
                            print("step {} / {}".format(e + 1, num_train_epochs))
                            train_batch = train_data[e]
                            train_input_batch = train_batch[0]
                            train_target_batch = train_batch[1]
                            sess.run(train_op, feed_dict={feature_ph: train_input_batch,
                                                          label_ph: train_target_batch})

                        sum_accuracy = 0
                        for e in range(num_eval_epochs):
                            print("evaluation eval {} / {}".format(e + 1, num_eval_epochs))
                            eval_batch = eval_data[e]
                            eval_input_batch = eval_batch[0]
                            eval_target_batch = eval_batch[1]
                            eval_batch_accuracy = sess.run(eval_op, feed_dict={feature_ph: eval_input_batch,
                                                                               label_ph: eval_target_batch})
                            sum_accuracy += eval_batch_accuracy

                        cur_accuracy = sum_accuracy / num_eval_epochs

                        end_time = timer()
                        dur_time = end_time - start_time

                        pre_accuracy = job_accuracy_dict[model_name]

                        job_time_dict[model_name] += dur_time
                        job_accuracy_dict[model_name] = cur_accuracy

                        if job_data['goal_type'] == 'accuracy':
                            if cur_accuracy >= job_data['goal_value']:
                                job_done_dict[model_name] = True
                                break
                        elif job_data['goal_type'] == 'runtime':
                            if job_time_dict[model_name] > job_data['goal_value']:
                                job_done_dict[model_name] = True
                                break
                        elif job_data['goal_type'] == 'convergence':
                            delta = cur_accuracy - pre_accuracy
                            if delta < job_data['goal_value']:
                                job_done_dict[model_name] = True
                                break
                        else:
                            raise ValueError('the job objective type is not supported')


if __name__ == "__main__":
    n_gpu = 2
    time_window = 30

    wg = WorkloadGenerator(cfg_workload.workload_size,
                           cfg_workload.cv_light_ratio,
                           cfg_workload.cv_med_ratio,
                           cfg_workload.cv_heavy_ratio,
                           cfg_workload.nlp_light_ratio,
                           cfg_workload.nlp_med_ratio,
                           cfg_workload.nlp_heavy_ratio,
                           cfg_workload.convergence_ratio,
                           cfg_workload.accuracy_ratio,
                           cfg_workload.runtime_ratio,
                           cfg_workload.random_seed)

    ml_workload = wg.generate_workload()

    job_queue = Manager().Queue()
    job_accuracy_dict = Manager().dict()
    job_time_dict = Manager().dict()
    job_done_dict = Manager().dict()

    proc_pool = Pool(processes=n_gpu)

    for job in ml_workload:
        job_key = str(job['id']) + '-' + job['model']
        job_queue.put(job)
        job_accuracy_dict[job_key] = 0
        job_time_dict[job_key] = 0
        job_done_dict[job_key] = False

    while not job_queue.empty():
        proc_pool.apply_async(func=train_job, args=(n_gpu, time_window))

    proc_pool.close()
    proc_pool.join()

    print("finish training workload")
