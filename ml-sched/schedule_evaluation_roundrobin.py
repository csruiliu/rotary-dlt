import tensorflow as tf
from multiprocessing import Process, Queue, Value

import config_parameter as cfg_para_yml
import config_path as cfg_path_yml
from utils_workload_func import generate_workload


def run_single_job_gpu(model_type, model_instance, batch_size,  assign_device, proc_stop):
    with tf.device(assign_device):
        features = tf.placeholder(tf.float32, [None, _img_width, _img_height, _img_channels])
        labels = tf.placeholder(tf.int64, [None, _img_num_class])


def run_single_job_cpu(assign_device):
    with tf.device(assign_device):
        pass




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

    _sch_wl = generate_workload(_sch_job_num, _sch_model_type_set, _sch_batch_size_set, _sch_optimizer_set,
                                _sch_learning_rate_set, _sch_activation_set, _sch_train_dataset)

    ##################################################
    # Training Dataset Parameter
    ##################################################

    if _sch_train_dataset == 'imagenet':
        _img_width = cfg_para_yml.img_width_imagenet
        _img_height = cfg_para_yml.img_height_imagenet
        _img_num_class = cfg_para_yml.num_class_imagenet
        _img_channels = cfg_para_yml.num_channels_rgb
        _img_path_train_data = cfg_path_yml.imagenet_t10k_img_raw_path
        _img_path_train_label = cfg_path_yml.imagenet_t10k_label_path
        _img_path_test_data = cfg_path_yml.imagenet_t1k_img_raw_path
        _img_path_test_label = cfg_path_yml.imagenet_t1k_label_path
    elif _sch_train_dataset == 'cifar10':
        _img_width = cfg_para_yml.img_width_cifar10
        _img_height = cfg_para_yml.img_height_cifar10
        _img_num_class = cfg_para_yml.num_class_cifar10
        _img_channels = cfg_para_yml.num_channels_rgb
        _img_path = cfg_path_yml.cifar_10_path
    elif _sch_train_dataset == 'mnist':
        _img_width = cfg_para_yml.img_width_mnist
        _img_height = cfg_para_yml.img_height_mnist
        _img_num_class = cfg_para_yml.num_class_imagenet
        _img_channels = cfg_para_yml.num_channels_bw
        _img_path_train_data = cfg_path_yml.mnist_train_img_path
        _img_path_train_label = cfg_path_yml.mnist_train_label_path
        _img_path_test_data = cfg_path_yml.mnist_test_10k_img_path
        _img_path_test_label = cfg_path_yml.mnist_test_10k_label_path
    else:
        raise ValueError('Only support dataset: imagenet, cifar10, mnist')

    ##################################################
    # Schedule Parameter
    ##################################################

    _sch_gpu_device_num = cfg_para_yml.sch_gpu_num
    _sch_cpu_device_num = cfg_para_yml.sch_cpu_num
    _sch_time_slots_num = cfg_para_yml.sch_time_slots_num
    _sch_slot_time_period = cfg_para_yml.sch_slot_time_period

    ##################################################
    # Round Robin
    ##################################################

    training_job_queue = Queue(_sch_gpu_device_num + _sch_cpu_device_num)



