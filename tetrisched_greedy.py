import tensorflow as tf
from multiprocessing import Process, Manager
from timeit import default_timer as timer
import os

import relish.config.config_parameter as cfg_para
import relish.config.config_path as cfg_path
from relish.common.model_importer import ModelImporter
from relish.common.dataset_loader import load_dataset_para, load_train_dataset, load_eval_dataset
from relish.tools.img_tool import load_imagenet_raw
from relish.tools.workload_func import generate_workload_slo


def schedule_job_roundrobin(sch_device_num, workload_use, workload_origin):
    sch_list = list()
    for i in range(sch_device_num):
        cur_job = workload_use.pop()
        if len(workload_use) == 0:
            for idx, value in enumerate(workload_origin):
                workload_use.append(value)
        sch_list.append(cur_job)

    return sch_list


def schedule_job_tetrisched(sch_device_num, acc_increment_dict, workload_origin):
    sch_list = list()
    selected_jobs = sorted(acc_increment_dict.items(),
                           key=lambda item: item[1], reverse=True)[:sch_device_num]

    for sjob in selected_jobs:
        sjob_id = int(sjob[0].split('_')[0])
        sch_list.append(workload_origin[sjob_id])

    return sch_list


def build_model(job_data, feature, label):
    train_dataset = job_data['train_dataset']
    img_w, img_h, num_chn, num_cls = load_dataset_para(train_dataset)

    train_model = ModelImporter(job_data['model_type'],
                                str(job_data['job_id']),
                                job_data['model_layer_num'],
                                img_h,
                                img_w,
                                num_chn,
                                num_cls,
                                job_data['batch_size'],
                                job_data['optimizer'],
                                job_data['learning_rate'],
                                job_data['activation'],
                                batch_padding=False)

    model_entity = train_model.get_model_entity()
    model_logit = model_entity.build(feature, is_training=True)
    model_train_op = model_entity.train(model_logit, label)
    model_eval_op = model_entity.evaluate(model_logit, label)

    model_name = '{}_{}_{}_{}_{}_{}_{}_{}'.format(job_data['job_id'], job_data['model_type'],
                                                  job_data['model_layer_num'], job_data['batch_size'],
                                                  job_data['optimizer'], job_data['learning_rate'],
                                                  job_data['activation'], job_data['train_dataset'])

    return model_train_op, model_eval_op, model_name


def train_job(job_data, job_progress_dict, assign_device):
    start_time = timer()

    train_dataset = job_data['train_dataset']
    train_feature_input, train_label_input = load_train_dataset(train_dataset)

    time_slots_num = cfg_para.sch_time_slots_num
    job_num = cfg_para.slo_job_num
    slot_time_period = cfg_para.sch_slot_time_period
    ckpt_save_path = cfg_path.ckpt_save_path + '/workload_' + str(job_num) + '_timeslot_' + str(time_slots_num)

    img_w, img_h, num_chn, num_cls = load_dataset_para(train_dataset)

    with tf.device(assign_device):
        features = tf.placeholder(tf.float32, [None, img_w, img_h, num_chn])
        labels = tf.placeholder(tf.int64, [None, num_cls])
        train_ops, _, model_name = build_model(job_data, features, labels)
        saver = tf.train.Saver()

        model_ckpt_save_path = ckpt_save_path + '/' + model_name
        if not os.path.exists(model_ckpt_save_path):
            os.makedirs(model_ckpt_save_path)

        checkpoint_file = model_ckpt_save_path + '/' + 'model_ckpt'
        train_batchsize = job_data['batch_size']

        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.gpu_options.allow_growth = True

        if train_dataset == 'imagenet':
            train_data_list = sorted(os.listdir(train_feature_input))

        with tf.Session(config=config) as sess:
            if os.path.isfile(checkpoint_file + '.meta'):
                saver.restore(sess, checkpoint_file)
            else:
                sess.run(tf.global_variables_initializer())

            num_batch = train_label_input.shape[0] // train_batchsize
            total_step = 0
            while True:
                for i in range(num_batch):
                    #print('step {} / {}'.format(i + 1, num_batch))
                    batch_offset = i * train_batchsize
                    batch_end = (i + 1) * train_batchsize

                    if train_dataset == 'imagenet':
                        batch_list = train_data_list[batch_offset:batch_end]
                        train_data_batch = load_imagenet_raw(train_feature_input,
                                                             batch_list,
                                                             img_h,
                                                             img_w)
                    else:
                        train_data_batch = train_feature_input[batch_offset:batch_end]

                    train_label_batch = train_label_input[batch_offset:batch_end]

                    sess.run(train_ops, feed_dict={features: train_data_batch, labels: train_label_batch})
                    total_step += 1
                    end_time = timer()
                    dur_time = end_time - start_time
                    if dur_time > slot_time_period:
                        job_progress_dict[model_name] += total_step
                        saver.save(sess, checkpoint_file)
                        return


def evaluate_job(job_data):
    time_slots_num = cfg_para.sch_time_slots_num
    job_num = cfg_para.slo_job_num
    ckpt_save_path = cfg_path.ckpt_save_path + '/workload_' + str(job_num) + '_timeslot_' + str(time_slots_num)

    train_dataset = job_data['train_dataset']

    img_w, img_h, num_chn, num_cls = load_dataset_para(train_dataset)
    eval_feature_input, eval_label_input = load_eval_dataset(train_dataset)

    graph = tf.Graph()
    with graph.as_default():
        features = tf.placeholder(tf.float32, [None, img_w, img_h, num_chn])
        labels = tf.placeholder(tf.int64, [None, num_cls])
        _, eval_ops, model_name = build_model(job_data, features, labels)
        saver = tf.train.Saver()

    model_ckpt_save_path = ckpt_save_path + '/' + model_name
    checkpoint_file = os.path.join(model_ckpt_save_path, 'model_ckpt')
    train_batchsize = job_data['batch_size']

    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True

    with tf.Session(graph=graph, config=config) as sess:
        if os.path.isfile(checkpoint_file + '.meta'):
            saver.restore(sess, checkpoint_file)
        else:
            sess.run(tf.global_variables_initializer())

        if train_dataset == 'imagenet':
            acc_sum = 0
            num_eval_batch = eval_label_input.shape[0] // 50
            eval_data_list = sorted(os.listdir(eval_feature_input))
            for n in range(num_eval_batch):
                batch_offset = n * train_batchsize
                batch_end = (n + 1) * train_batchsize
                batch_eval_list = eval_data_list[batch_offset:batch_end]
                feature_eval_batch = load_imagenet_raw(eval_feature_input, batch_eval_list, img_h, img_w)
                label_eval_batch = eval_label_input[batch_offset:batch_end]
                acc_batch = sess.run(eval_ops, feed_dict={features: feature_eval_batch, labels: label_eval_batch})
                acc_sum += acc_batch

            model_acc_avg = acc_sum / num_eval_batch
        else:
            model_acc_avg = sess.run(eval_ops, feed_dict={features: eval_feature_input,
                                                          labels: eval_label_input})

    return model_acc_avg, model_name


def evaluate_job_during_run(job_data, job_accuracy_increment_dict, job_current_accuracy_dict, assign_device):
    train_dataset = job_data['train_dataset']
    time_slots_num = cfg_para.sch_time_slots_num
    job_num = cfg_para.slo_job_num
    img_w, img_h, num_chn, num_cls = load_dataset_para(train_dataset)
    eval_feature_input, eval_label_input = load_eval_dataset(train_dataset)

    ckpt_save_path = cfg_path.ckpt_save_path + '/workload_' + str(job_num) + '_timeslot_' + str(time_slots_num)

    with tf.device(assign_device):
        graph = tf.Graph()
        with graph.as_default():
            features = tf.placeholder(tf.float32, [None, img_w, img_h, num_chn])
            labels = tf.placeholder(tf.int64, [None, num_cls])
            _, eval_ops, model_name = build_model(job_data, features, labels)
            saver = tf.train.Saver()

        model_ckpt_save_path = ckpt_save_path + '/' + model_name
        checkpoint_file = os.path.join(model_ckpt_save_path, 'model_ckpt')
        train_batchsize = job_data['batch_size']

        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.gpu_options.allow_growth = True

        with tf.Session(graph=graph, config=config) as sess:
            if os.path.isfile(checkpoint_file + '.meta'):
                saver.restore(sess, checkpoint_file)
            else:
                sess.run(tf.global_variables_initializer())

            if train_dataset == 'imagenet':
                acc_sum = 0
                num_eval_batch = eval_label_input.shape[0] // 50
                eval_data_list = sorted(os.listdir(eval_feature_input))
                for n in range(num_eval_batch):
                    batch_offset = n * train_batchsize
                    batch_end = (n + 1) * train_batchsize
                    batch_eval_list = eval_data_list[batch_offset:batch_end]
                    feature_eval_batch = load_imagenet_raw(eval_feature_input, batch_eval_list, img_h, img_w)
                    label_eval_batch = eval_label_input[batch_offset:batch_end]
                    acc_batch = sess.run(eval_ops, feed_dict={features: feature_eval_batch,
                                                              labels: label_eval_batch})
                    acc_sum += acc_batch

                model_acc_avg = acc_sum / num_eval_batch
            else:
                model_acc_avg = sess.run(eval_ops, feed_dict={features: eval_feature_input,
                                                              labels: eval_label_input})

    job_accuracy_increment_dict[model_name] = model_acc_avg - job_current_accuracy_dict[model_name]
    job_current_accuracy_dict[model_name] = model_acc_avg


def tetrisched_run():
    ##################################################
    # Key parameters
    ##################################################
    time_slots_num = cfg_para.sch_time_slots_num
    job_num = cfg_para.slo_job_num

    ##################################################
    # Generate workload
    ##################################################

    sched_workload = generate_workload_slo(job_num, use_seed=True)
    sched_workload_use = sched_workload.copy()

    ##################################################
    # Prepare the shared dict
    ##################################################

    # record the progress of each job during a specific schedule
    job_progress_dict = Manager().dict()

    # record the progress of each job during a specific schedule
    job_accuracy_increment_dict = Manager().dict()
    job_current_accuracy_dict = Manager().dict()

    for job in sched_workload:
        model_name = '{}_{}_{}_{}_{}_{}_{}_{}'.format(job['job_id'], job['model_type'],
                                                      job['model_layer_num'], job['batch_size'],
                                                      job['optimizer'], job['learning_rate'],
                                                      job['activation'], job['train_dataset'])
        job_progress_dict[model_name] = 0
        job_accuracy_increment_dict[model_name] = 0
        job_current_accuracy_dict[model_name] = 0

    ##################################################
    # Schedule Parameter
    ##################################################

    gpu_device_num = cfg_para.sch_gpu_num
    cpu_device_num = cfg_para.sch_cpu_num
    sched_device_num = gpu_device_num + cpu_device_num
    slot_time_period = cfg_para.sch_slot_time_period

    ##################################################
    # TetriSched with Greedy Mechanism
    ##################################################

    is_cover_workload = False

    time_slot_count = 0
    while time_slot_count < time_slots_num:
        print('current time slot {}'.format(time_slot_count))
        if is_cover_workload:
            job_list = schedule_job_tetrisched(sched_device_num, job_accuracy_increment_dict, sched_workload)
        else:
            job_list = schedule_job_roundrobin(sched_device_num, sched_workload_use, sched_workload)

        # Run Job in TetriSched
        proc_gpu_list = list()
        for gn in range(gpu_device_num):
            assign_gpu = '/gpu:' + str(gn)
            proc_gpu = Process(target=train_job, args=(job_list[gn], job_progress_dict, assign_gpu))
            proc_gpu_list.append(proc_gpu)
        proc_cpu = Process(target=train_job, args=(job_list[sched_device_num - 1], job_progress_dict, '/cpu:0'))

        for proc_gpu in proc_gpu_list:
            proc_gpu.start()
        proc_cpu.start()

        for proc_gpu in proc_gpu_list:
            proc_gpu.join()
        proc_cpu.join()

        # Evaluate Job in TetriSched
        if is_cover_workload:
            proc_gpu_eval_list = list()
            for gn in range(gpu_device_num):
                assign_gpu = '/gpu:' + str(gn)
                proc_gpu = Process(target=evaluate_job_during_run, args=(job_list[gn], job_accuracy_increment_dict,
                                                                         job_current_accuracy_dict, assign_gpu))
                proc_gpu_list.append(proc_gpu)
            proc_cpu_eval = Process(target=evaluate_job_during_run, args=(job_list[sched_device_num - 1],
                                                                          job_accuracy_increment_dict,
                                                                          job_current_accuracy_dict, '/cpu:0'))

            for proc_gpu_eval in proc_gpu_eval_list:
                proc_gpu_eval.start()
            proc_cpu_eval.start()

            for proc_gpu_eval in proc_gpu_eval_list:
                proc_gpu_eval.join()
            proc_cpu_eval.join()

        time_slot_count += 1

    ##################################################
    # Evaluate workload after TetriSched schedule
    ##################################################

    sch_job_attainment_list = list()
    sch_job_name_list = list()
    sch_job_progress_list = list()

    for jidx in sched_workload:
        job_accuracy, job_name = evaluate_job(jidx)
        sch_job_attainment_list.append(job_accuracy)
        sch_job_name_list.append(job_name)
        sch_job_progress_list.append(job_progress_dict[job_name])

    workload_acc_avg = sum(sch_job_attainment_list) / job_num

    print('#########################################################')
    print('jobs attainment in the workload:')
    for job_idx, _ in enumerate(sched_workload):
        print('**Job Result**: {}_{}_{}'.format(sch_job_name_list[job_idx],
                                                sch_job_attainment_list[job_idx],
                                                sch_job_progress_list[job_idx]))
    print('**Workload Result**: {}'.format(workload_acc_avg))
