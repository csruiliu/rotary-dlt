import tensorflow as tf
from multiprocessing import Process
from timeit import default_timer as timer
import os

from model_importer import ModelImporter
import config_parameter as cfg_para_yml
import config_path as cfg_path_yml
from utils_workload_func import generate_workload
from utils_img_func import load_imagenet_raw, load_imagenet_labels_onehot, load_cifar10_keras, load_mnist_image, load_mnist_label_onehot


def produce_job_roundrobin():
    cur_job = _sch_workload_use.pop()
    if len(_sch_workload_use) == 0:
        for idx, value in enumerate(_sch_workload):
            _sch_workload_use.append(value)
    return cur_job


def schedule_job_roundrobin():
    sch_list = list()
    for i in range(_sch_device_num):
        job = produce_job_roundrobin()
        sch_list.append(job)
    return sch_list


def run_job(job_info, assign_device):
    start_time = timer()
    with tf.device(assign_device):
        features = tf.placeholder(tf.float32, [None, _img_width, _img_height, _img_channels])
        labels = tf.placeholder(tf.int64, [None, _img_num_class])

        train_batchsize = job_info['batch_size']

        train_model = ModelImporter(job_info['model_type'], job_info['job_id'], job_info['model_layer_num'],
                                    _img_height, _img_width, _img_channels, _img_num_class, train_batchsize,
                                    job_info['optimizer'], job_info['learning_rate'], job_info['activation'], False)

        model_entity = train_model.get_model_entity()
        model_logit = model_entity.build(features, is_training=True)
        train_ops = model_entity.train(model_logit, labels)

        model_name = '{0}-{1}-{2}-{3}-{4}-{5}-{6}-{7}'.format(job_info['job_id'], job_info['model_type'],
                                                              job_info['model_layer_num'], train_batchsize,
                                                              job_info['optimizer'], job_info['learning_rate'],
                                                              job_info['activation'], job_info['train_dataset'])

        model_ckpt_save_path = _ckpt_save_path + '/' + model_name

        saver = tf.train.Saver()

        with tf.device(assign_device):
            config = tf.ConfigProto()
            config.allow_soft_placement = True
            config.gpu_options.allow_growth = True

            if _sch_train_dataset == 'imagenet':
                train_data_list = sorted(os.listdir(_imagenet_train_data_path))

            with tf.Session(config=config) as sess:
                if os.path.exists(model_ckpt_save_path):
                    saver.restore(sess, model_ckpt_save_path)
                else:
                    sess.run(tf.global_variables_initializer())

                num_batch = train_label.shape[0] // train_batchsize

                while True:
                    for i in range(num_batch):
                        print('step {} / {}'.format(i + 1, num_batch))
                        batch_offset = i * train_batchsize
                        batch_end = (i + 1) * train_batchsize

                        if train_data == 'imagenet':
                            batch_list = train_data_list[batch_offset:batch_end]
                            train_data_batch = load_imagenet_raw(_imagenet_train_data_path, batch_list, _img_height,
                                                                 _img_width)
                        else:
                            train_data_batch = train_data[batch_offset:batch_end]

                        train_label_batch = train_label[batch_offset:batch_end]

                        sess.run(train_ops, feed_dict={features: train_data_batch, labels: train_label_batch})

                        end_time = timer()
                        dur_time = end_time - start_time
                        if dur_time > _sch_slot_time_period:
                            saver.save(sess, model_ckpt_save_path)

                            return


if __name__ == "__main__":
    ##################################################
    # Generate Workload
    ##################################################

    _sch_job_num = cfg_para_yml.sch_job_num
    _sch_model_type_set = cfg_para_yml.sch_model_type_set
    _sch_batch_size_set = cfg_para_yml.sch_batch_size_set
    _sch_optimizer_set = cfg_para_yml.sch_optimizer_set
    _sch_learning_rate_set = cfg_para_yml.sch_learning_rate_set
    _sch_activation_set = cfg_para_yml.sch_activation_set
    _sch_train_dataset = cfg_para_yml.train_dataset

    _sch_workload = generate_workload(_sch_job_num, _sch_model_type_set, _sch_batch_size_set, _sch_optimizer_set,
                                      _sch_learning_rate_set, _sch_activation_set, _sch_train_dataset)
    _sch_workload_use = _sch_workload.copy()

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
        _imagenet_test_data_path = cfg_path_yml.imagenet_t1k_img_raw_path
        _imagenet_test_label_path = cfg_path_yml.imagenet_t1k_label_path

        train_label = load_imagenet_labels_onehot(_imagenet_train_label_path, _num_classes)
        eval_label = load_imagenet_labels_onehot(_imagenet_test_label_path, _num_classes)

    elif _sch_train_dataset == 'cifar10':
        _img_width = cfg_para_yml.img_width_cifar10
        _img_height = cfg_para_yml.img_height_cifar10
        _img_num_class = cfg_para_yml.num_class_cifar10
        _img_channels = cfg_para_yml.num_channels_rgb
        _img_path = cfg_path_yml.cifar_10_path

        cifar10_path = cfg_path_yml.cifar_10_path
        train_data, train_label, test_data, test_label = load_cifar10_keras()

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
    _sch_time_slots_num = cfg_para_yml.sch_time_slots_num
    _sch_slot_time_period = cfg_para_yml.sch_slot_time_period
    _ckpt_save_path = cfg_path_yml.ckpt_save_path

    ##################################################
    # Round Robin Schedule
    ##################################################
    proc_gpu_list = list()

    time_slot_count = 0
    while time_slot_count < _sch_time_slots_num:
        print('current time slot {}'.format(time_slot_count))
        job_list = schedule_job_roundrobin()
        proc_gpu_list = list()
        for gn in range(_sch_gpu_device_num):
            assign_gpu = '/gpu:' + str(gn)
            device_proc_gpu = Process(target=run_job, args=(job_list[gn], assign_gpu))
            proc_gpu_list.append(device_proc_gpu)

        device_proc_cpu = Process(target=run_job, args=(job_list[_sch_device_num-1], '/cpu:0'))

        for device_proc_gpu in proc_gpu_list:
            device_proc_gpu.start()

        device_proc_cpu.start()

        for device_proc_gpu in proc_gpu_list:
            device_proc_gpu.join()
        device_proc_cpu.join()

        time_slot_count += 1
