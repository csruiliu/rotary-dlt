import tensorflow as tf
from multiprocessing import Process, Manager
import argparse
from timeit import default_timer as timer
import os

import relish.config.config_parameter as cfg_para_yml
import relish.config.config_path as cfg_path_yml
from relish.core.relish_environment import SchedEnv
from relish.core.relish_engine import SchedEngine
from relish.estimator.estimator_model_accuracy import AccuracyEstimator
from relish.estimator.estimator_model_steptime_multidevices import MultiDeviceTimeEstimator
from relish.models.model_importer import ModelImporter
from relish.tools.utils_workload_func import generate_workload_slo
from relish.tools.utils_img_func import load_imagenet_raw, load_imagenet_labels_onehot, load_cifar10_keras, load_mnist_image, load_mnist_label_onehot

tf.compat.v1.enable_v2_behavior()


def produce_job_roundrobin():
    global _is_cover_workload
    if len(_sch_workload_use) != 0:
        cur_job = _sch_workload_use.pop()
        if len(_sch_workload_use) == 0:
            _is_cover_workload = True
        return cur_job


def schedule_job_roundrobin():
    sch_list = list()
    for i in range(_sch_device_num):
        job = produce_job_roundrobin()
        sch_list.append(job)
    return sch_list


def build_model(job_data, ph_features, ph_labels):
    train_model = ModelImporter(job_data['model_type'], str(job_data['job_id']), job_data['model_layer_num'],
                                _img_height, _img_width, _img_channels, _img_num_class, job_data['batch_size'],
                                job_data['optimizer'], job_data['learning_rate'], job_data['activation'], False)

    model_entity = train_model.get_model_entity()
    model_logit = model_entity.build(ph_features, is_training=True)
    model_train_op = model_entity.train(model_logit, ph_labels)
    model_eval_op = model_entity.evaluate(model_logit, ph_labels)

    model_name = '{}_{}_{}_{}_{}_{}_{}_{}'.format(job_data['job_id'], job_data['model_type'],
                                                  job_data['model_layer_num'], job_data['batch_size'],
                                                  job_data['optimizer'], job_data['learning_rate'],
                                                  job_data['activation'], job_data['train_dataset'])

    return model_train_op, model_eval_op, model_name


def run_job(job_info, job_progress_dict, assign_device):
    start_time = timer()

    with tf.device(assign_device):
        features = tf.placeholder(tf.float32, [None, _img_width, _img_height, _img_channels])
        labels = tf.placeholder(tf.int64, [None, _img_num_class])
        train_ops, _, model_name = build_model(job_info, features, labels)
        saver = tf.train.Saver()

        model_ckpt_save_path = _ckpt_save_path + '/' + model_name
        if not os.path.exists(model_ckpt_save_path):
            os.makedirs(model_ckpt_save_path)

        checkpoint_file = model_ckpt_save_path + '/' + 'model_ckpt'
        train_batchsize = job_info['batch_size']

        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.gpu_options.allow_growth = True

        if _sch_train_dataset == 'imagenet':
            train_data_list = sorted(os.listdir(_imagenet_train_data_path))

        with tf.Session(config=config) as sess:
            if os.path.isfile(checkpoint_file + '.meta'):
                saver.restore(sess, checkpoint_file)
            else:
                sess.run(tf.global_variables_initializer())

            num_batch = train_label.shape[0] // train_batchsize
            total_step = 0
            while True:
                for i in range(num_batch):
                    #print('step {} / {}'.format(i + 1, num_batch))
                    batch_offset = i * train_batchsize
                    batch_end = (i + 1) * train_batchsize

                    if _sch_train_dataset == 'imagenet':
                        batch_list = train_data_list[batch_offset:batch_end]
                        train_data_batch = load_imagenet_raw(_imagenet_train_data_path, batch_list, _img_height,
                                                             _img_width)
                    else:
                        train_data_batch = train_data[batch_offset:batch_end]

                    train_label_batch = train_label[batch_offset:batch_end]

                    sess.run(train_ops, feed_dict={features: train_data_batch, labels: train_label_batch})
                    total_step += 1
                    end_time = timer()
                    dur_time = end_time - start_time
                    if dur_time > _sch_slot_time_period:
                        job_progress_dict[model_name] += total_step
                        saver.save(sess, checkpoint_file)
                        return


def schedule_job_rlsched():
    # init schedule environment
    mlsch_env = SchedEnv(time_slots_num=_sch_time_slots_num, gpu_device_num=_sch_gpu_device_num,
                         cpu_device_num=_sch_cpu_device_num, workload=_sch_workload,
                         reward_function=_sch_reward_function, is_simulation=True)

    # inti schedule multi-device time estimator
    mlsch_mte = MultiDeviceTimeEstimator(top_k=3)
    mlsch_mte.import_steptime_dataset(_steptime_dataset_path)

    # inti schedule multi-device accuracy estimator
    mlsch_ae = AccuracyEstimator(top_k=3)
    mlsch_ae.import_accuracy_dataset(_accuracy_dataset_path)

    mlsch_env.load_estimator(mlsch_mte, mlsch_ae)

    mlsch_engine = SchedEngine(mlsch_env)

    mlsch_engine.build_sch_agent()
    mlsch_engine.benchmark_before_training(benchmark_num_episodes=20)

    mlsch_engine.train_sch_agent(num_train_iterations=50, collect_episodes_per_iteration=5, steps_num_per_batch=100,
                                 log_interval=25, include_eval=False, eval_interval=50, num_eval_episodes_for_train=15)

    final_reward = mlsch_engine.evaluate_sch_agent(eval_num_episodes=20)
    print('final reward: {}'.format(final_reward))
    sch_list = mlsch_engine.generate_schedule()
    return sch_list


def evaluate_job(job_info, job_accuracy_increment_dict, job_current_accuracy_dict):
    graph = tf.Graph()
    with graph.as_default():
        features = tf.placeholder(tf.float32, [None, _img_width, _img_height, _img_channels])
        labels = tf.placeholder(tf.int64, [None, _img_num_class])
        _, eval_ops, model_name = build_model(job_info, features, labels)
        saver = tf.train.Saver()

    model_ckpt_save_path = _ckpt_save_path + '/' + model_name
    checkpoint_file = os.path.join(model_ckpt_save_path, 'model_ckpt')
    train_batchsize = job_info['batch_size']

    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True

    with tf.Session(graph=graph, config=config) as sess:
        if os.path.isfile(checkpoint_file + '.meta'):
            saver.restore(sess, checkpoint_file)
        else:
            sess.run(tf.global_variables_initializer())

        if _sch_train_dataset == 'imagenet':
            acc_sum = 0
            num_eval_batch = train_label.shape[0] // 50
            for n in range(num_eval_batch):
                batch_offset = n * train_batchsize
                batch_end = (n + 1) * train_batchsize
                batch_eval_list = eval_data[batch_offset:batch_end]
                feature_eval_batch = load_imagenet_raw(_imagenet_eval_data_path, batch_eval_list, _img_height, _img_width)
                label_eval_batch = eval_label[batch_offset:batch_end]
                acc_batch = sess.run(eval_ops, feed_dict={features: feature_eval_batch, labels: label_eval_batch})
                acc_sum += acc_batch

            model_acc_avg = acc_sum / num_eval_batch
        else:
            model_acc_avg = sess.run(eval_ops, feed_dict={features: eval_data, labels: eval_label})

    job_accuracy_increment_dict[model_name] = model_acc_avg - job_current_accuracy_dict[model_name]
    job_current_accuracy_dict[model_name] = model_acc_avg

    return model_acc_avg, model_name


def init_shared_dict():
    for job in _sch_workload:
        model_name = '{}_{}_{}_{}_{}_{}_{}_{}'.format(job['job_id'], job['model_type'],
                                                      job['model_layer_num'], job['batch_size'],
                                                      job['optimizer'], job['learning_rate'],
                                                      job['activation'], job['train_dataset'])
        _sch_job_progress_dict[model_name] = 0
        _job_accuracy_increment_dict[model_name] = 0
        _job_current_accuracy_dict[model_name] = 0


if __name__ == "__main__":
    ##################################################
    # Key Parameters for evaluation
    ##################################################
    parser = argparse.ArgumentParser()

    parser.add_argument('-j', '--job_num', action='store', type=int, help='the number of jobs in a workload')
    parser.add_argument('-t', '--time_slot', action='store', type=int, help='the number of time slots')
    args = parser.parse_args()

    if args.job_num is not None:
        _sch_job_num = args.job_num
    else:
        _sch_job_num = cfg_para_yml.slo_job_num

    if args.time_slot is not None:
        _sch_time_slots_num = args.time_slot
    else:
        _sch_time_slots_num = cfg_para_yml.sch_time_slots_num

    ##################################################
    # Generate Workload
    ##################################################

    _sch_model_type_set = cfg_para_yml.slo_model_type_set
    _sch_batch_size_set = cfg_para_yml.slo_batch_size_set
    _sch_optimizer_set = cfg_para_yml.slo_optimizer_set
    _sch_learning_rate_set = cfg_para_yml.slo_learning_rate_set
    _sch_activation_set = cfg_para_yml.slo_activation_set
    _sch_train_dataset = cfg_para_yml.train_dataset

    _sch_workload = generate_workload_slo(_sch_job_num, _sch_model_type_set, _sch_batch_size_set, _sch_optimizer_set,
                                          _sch_learning_rate_set, _sch_activation_set, _sch_train_dataset, True)
    _sch_workload_use = _sch_workload.copy()

    ##################################################
    # Prepare the shared dict
    ##################################################

    # record the progress of each job during a specific schedule
    _sch_job_progress_dict = Manager().dict()

    # record the progress of each job during a specific schedule
    _job_accuracy_increment_dict = Manager().dict()
    _job_current_accuracy_dict = Manager().dict()

    init_shared_dict()

    ##################################################
    # Prepare Training Dataset
    ##################################################

    if _sch_train_dataset == 'imagenet':
        _img_width = cfg_para_yml.img_width_imagenet
        _img_height = cfg_para_yml.img_height_imagenet
        _num_classes = cfg_para_yml.num_class_imagenet
        _img_channels = cfg_para_yml.num_channels_rgb
        _imagenet_train_data_path = cfg_path_yml.imagenet_t10k_img_raw_path
        _imagenet_train_label_path = cfg_path_yml.imagenet_t10k_label_path
        _imagenet_eval_data_path = cfg_path_yml.imagenet_t1k_img_raw_path
        _imagenet_eval_label_path = cfg_path_yml.imagenet_t1k_label_path

        train_label = load_imagenet_labels_onehot(_imagenet_train_label_path, _num_classes)
        eval_label = load_imagenet_labels_onehot(_imagenet_eval_label_path, _num_classes)

    elif _sch_train_dataset == 'cifar10':
        _img_width = cfg_para_yml.img_width_cifar10
        _img_height = cfg_para_yml.img_height_cifar10
        _img_num_class = cfg_para_yml.num_class_cifar10
        _img_channels = cfg_para_yml.num_channels_rgb
        _img_path = cfg_path_yml.cifar_10_path

        cifar10_path = cfg_path_yml.cifar_10_path
        train_data, train_label, eval_data, eval_label = load_cifar10_keras()

    elif _sch_train_dataset == 'mnist':
        _img_width = cfg_para_yml.img_width_mnist
        _img_height = cfg_para_yml.img_height_mnist
        _img_num_class = cfg_para_yml.num_class_imagenet
        _img_channels = cfg_para_yml.num_channels_bw

        _mnist_train_img_path = cfg_path_yml.mnist_train_img_path
        _mnist_train_label_path = cfg_path_yml.mnist_train_label_path
        _mnist_test_img_path = cfg_path_yml.mnist_test_10k_img_path
        _mnist_test_label_path = cfg_path_yml.mnist_test_10k_label_path

        train_data = load_mnist_image(_mnist_train_img_path)
        train_label = load_mnist_label_onehot(_mnist_train_label_path)
        eval_data = load_mnist_image(_mnist_test_img_path)
        eval_label = load_mnist_label_onehot(_mnist_test_label_path)

    else:
        raise ValueError('Only support dataset: imagenet, cifar10, mnist')

    ##################################################
    # Schedule Parameter
    ##################################################

    _sch_gpu_device_num = cfg_para_yml.sch_gpu_num
    _sch_cpu_device_num = cfg_para_yml.sch_cpu_num
    _sch_device_num = _sch_gpu_device_num + _sch_cpu_device_num
    _sch_slot_time_period = cfg_para_yml.sch_slot_time_period
    _ckpt_save_path = cfg_path_yml.ckpt_save_path + '/workload_' + str(_sch_job_num) + '_timeslot_' + str(_sch_time_slots_num)

    # reward function
    _sch_reward_function = cfg_para_yml.slo_reward_function
    print("Reward Function: {}".format(_sch_reward_function))

    # Get path parameters from config
    _steptime_dataset_path = cfg_path_yml.multidevices_time_dataset_path
    _accuracy_dataset_path = cfg_path_yml.accuracy_dataset_path

    ##################################################
    # Reinforcement Learning Schedule
    ##################################################

    _is_cover_workload = False
    time_slot_count = 0
    while time_slot_count < _sch_time_slots_num:
        if _is_cover_workload:
            print('starting the rl-based scheduling')
            job_list = schedule_job_rlsched()
        else:
            job_list = schedule_job_roundrobin()

        proc_gpu_list = list()

        for gn in range(_sch_gpu_device_num):
            assign_gpu = '/gpu:' + str(gn)
            proc_gpu = Process(target=run_job, args=(job_list[gn], _sch_job_progress_dict, assign_gpu))
            proc_gpu_list.append(proc_gpu)
        proc_cpu = Process(target=run_job, args=(job_list[_sch_device_num - 1], _sch_job_progress_dict, '/cpu:0'))

        for proc_gpu in proc_gpu_list:
            proc_gpu.start()
        proc_cpu.start()

        time_slot_count += 1

    sch_job_attainment_list = list()
    sch_job_name_list = list()
    sch_job_progress_list = list()

    for jidx in _sch_workload:
        job_accuracy, job_name = evaluate_job(jidx)
        sch_job_attainment_list.append(job_accuracy)
        sch_job_name_list.append(job_name)
        sch_job_progress_list.append(_sch_job_progress_dict[job_name])

    workload_acc_avg = sum(sch_job_attainment_list) / _sch_job_num

    print('#########################################################')
    print('jobs attainment in the workload:')
    for job_idx, _ in enumerate(_sch_workload):
        print('**Job Result**: {}_{}_{}'.format(sch_job_name_list[job_idx], sch_job_attainment_list[job_idx],
                                                sch_job_progress_list[job_idx]))
    print('**Workload Result**: {}'.format(workload_acc_avg))