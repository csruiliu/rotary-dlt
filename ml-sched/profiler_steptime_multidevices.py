import tensorflow as tf
import argparse
import multiprocessing as mp
import numpy as np
from timeit import default_timer as timer
import os

import config_path as cfg_path_yml
import config_parameter as cfg_para_yml
from utils_img_func import load_imagenet_labels_onehot, load_imagenet_raw, load_cifar10_keras
from model_importer import ModelImporter


def generate_job_queue():
    cpu_job_queue = mp.Queue()
    gpu_job_queue = mp.Queue()

    workload_num = _cpu_model_num + _gpu_model_num
    _rand_seed = cfg_para_yml.rand_seed
    np.random.seed(_rand_seed)
    model_name_abbr = np.random.choice(_rand_seed, workload_num, replace=False).tolist()

    for _ in range(_gpu_model_num):
        if not gpu_job_queue.full():
            gpu_job_queue.put([_gpu_model_type, model_name_abbr.pop(), _gpu_model_layer, _gpu_batch_size,
                               _gpu_optimizer, _gpu_learn_rate, _gpu_activation])

    for _ in range(_cpu_model_num):
        if not cpu_job_queue.full():
            cpu_job_queue.put([_cpu_model_type, model_name_abbr.pop(), _cpu_model_layer, _cpu_batch_size,
                               _cpu_optimizer, _cpu_learn_rate, _cpu_activation])

    return gpu_job_queue, cpu_job_queue


def run_single_job(model_type, model_instance, layer_num, batch_size, optimizer, learning_rate, activation,
                   assign_device, proc_idx=0):
    with tf.device(assign_device):
        features = tf.placeholder(tf.float32, [None, _img_width, _img_height, _num_channels])
        labels = tf.placeholder(tf.int64, [None, _num_classes])

        train_model = ModelImporter(model_type, str(model_instance), layer_num, _img_height, _img_width, _num_channels,
                                    _num_classes, batch_size, optimizer, learning_rate, activation, False)

        model_entity = train_model.get_model_entity()
        model_logit = model_entity.build(features, is_training=True)
        train_ops = model_entity.train(model_logit, labels)

        num_conv_layer, num_pool_layer, num_residual_layer = model_entity.get_layer_info()

        model_arch_info = num_conv_layer + '-' + num_pool_layer + '-' + num_residual_layer




        if use_raw_image:
            image_list = sorted(os.listdir(_image_path_raw))

        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        tf_config.allow_soft_placement = True

        print('{} job start...'.format(assign_device))

        step_time = 0
        step_count = 0

        with tf.Session(config=tf_config) as sess:
            sess.run(tf.global_variables_initializer())
            num_batch = train_label.shape[0] // batch_size
            for i in range(num_batch):
                if assign_device.startswith('/cpu'):
                    print('**CPU JOB**: Proc-{}, {}-{}-{} on cpu [{}]: step {} / {}'.format(proc_idx, model_type,
                                                                                            batch_size, model_instance,
                                                                                            timer(), i + 1, num_batch))
                elif assign_device.startswith('/gpu'):
                    print('**GPU JOB**: {}-{}-{} on cpu [{}]: step {} / {}'.format(model_type, batch_size,
                                                                                   model_instance, timer(),
                                                                                   i + 1, num_batch))
                if i != 0:
                    start_time = timer()

                batch_offset = i * batch_size
                batch_end = (i + 1) * batch_size
                if use_raw_image:
                    batch_list = image_list[batch_offset:batch_end]
                    X_mini_batch_feed = load_imagenet_raw(_image_path_raw, batch_list, _img_height, _img_width)
                else:
                    X_mini_batch_feed = train_data[batch_offset:batch_end]
                Y_mini_batch_feed = train_label[batch_offset:batch_end]
                sess.run(train_ops, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})

                if i != 0:
                    end_time = timer()
                    dur_time = end_time - start_time
                    print("step time:", dur_time)
                    step_time += dur_time
                    step_count += 1
        if assign_device.startswith('/cpu'):
            print('CPU job average step time [{0}]: {1}'.format(timer(), step_time / step_count * 1000))
        elif assign_device.startswith('/gpu'):
            print('GPU job average step time [{0}]: {1}'.format(timer(), step_time / step_count * 1000))


def consumer_gpu(queue, assign_device):
    if not queue.empty():
        gpu_job = queue.get()
        p = mp.Process(target=run_single_job, args=(gpu_job[0], gpu_job[1], gpu_job[2], gpu_job[3], gpu_job[4],
                                                    gpu_job[5], gpu_job[6], assign_device))
        p.start()
        p.join()


def consumer_cpu(queue, assign_device):
    for proc_idx in range(os_thread_num):
        if not queue.empty():
            cpu_job = queue.get()
            p = mp.Process(target=run_single_job, args=(cpu_job[0], cpu_job[1], cpu_job[2], cpu_job[3], cpu_job[4],
                                                        cpu_job[5], cpu_job[6], assign_device, proc_idx))
            p.start()
        else:
            break


if __name__ == "__main__":

    ########################################
    # Get parameters from argument
    ########################################

    parser = argparse.ArgumentParser()
    parser.add_argument('-cm', '--cpu_model', required=True, action='store',
                        choices=['resnet', 'mobilenet', 'densenet', 'mlp', 'scn'],
                        help='cpu model type [resnet,mobilenet,mlp,densenet]')
    parser.add_argument('-cl', '--cpu_model_layer', required=True, action='store', type=int,
                        help='number layer of cpu model')
    parser.add_argument('-cn', '--cpu_model_num', required=True, action='store', type=int,
                        help='indicate the number of cpu model')
    parser.add_argument('-cb', '--cpu_batch_size', required=True, action='store', type=int,
                        help='indicate the batch size for cpu model')
    parser.add_argument('-cr', '--cpu_learn_rate', required=True, action='store', type=float,
                        help='learning rate like 0.1, 0.01, 0.001, 0.0001')
    parser.add_argument('-ca', '--cpu_activation', required=True, action='store', type=str,
                        help='activation function for cpu model, like relu, tanh')
    parser.add_argument('-co', '--cpu_optimizer', required=True, action='store', type=str,
                        help='optimizer for cpu model, like Adam, SGD, Momentum')

    parser.add_argument('-gm', '--gpu_model', required=True, action='store',
                        choices=['resnet', 'mobilenet', 'densenet', 'mlp', 'scn'],
                        help='gpu model type [resnet,mobilenet,mlp,densenet]')
    parser.add_argument('-gl', '--gpu_model_layer', required=True, action='store', type=int,
                        help='number layer of gpu model')
    parser.add_argument('-gn', '--gpu_model_num', required=True, action='store', type=int,
                        help='indicate the number of gpu model')
    parser.add_argument('-gb', '--gpu_batch_size', required=True, action='store', type=int,
                        help='indicate the batch size for gpu model')
    parser.add_argument('-gr', '--gpu_learn_rate', required=True, action='store', type=float,
                        help='learning rate like 0.1, 0.01, 0.001, 0.0001')
    parser.add_argument('-ga', '--gpu_activation', required=True, action='store', type=str,
                        help='activation function for gpu model like relu, tanh')
    parser.add_argument('-go', '--gpu_optimizer', required=True, action='store', type=str,
                        help='optimizer for gpu model, like Adam, SGD, Momentum')

    parser.add_argument('-t', '--train_dataset', required=True, action='store', choices=['imagenet', 'cifar10'],
                        help='training set [imagenet, cifar10]')

    args = parser.parse_args()

    _cpu_model_type = args.cpu_model
    _cpu_model_layer = args.cpu_model_layer
    _cpu_model_num = args.cpu_model_num
    _cpu_batch_size = args.cpu_batch_size
    _cpu_learn_rate = args.cpu_learn_rate
    _cpu_activation = args.cpu_activation
    _cpu_optimizer = args.cpu_optimizer

    _gpu_model_type = args.gpu_model
    _gpu_model_layer = args.gpu_model_layer
    _gpu_model_num = args.gpu_model_num
    _gpu_batch_size = args.gpu_batch_size
    _gpu_learn_rate = args.gpu_learn_rate
    _gpu_activation = args.gpu_activation
    _gpu_optimizer = args.gpu_optimizer

    _train_dataset = args.train_dataset

    ########################################
    # Get dataset parameters from config
    ########################################

    if _train_dataset == 'imagenet':
        _image_path_raw = cfg_path_yml.imagenet_t10k_img_raw_path
        _label_path = cfg_path_yml.imagenet_t1k_label_path
        _img_width = cfg_para_yml.img_width_imagenet
        _img_height = cfg_para_yml.img_height_imagenet
        _num_channels = cfg_para_yml.num_channels_rgb
        _num_classes = cfg_para_yml.num_class_imagenet
        train_label = load_imagenet_labels_onehot(_label_path, _num_classes)
        use_raw_image = True
    elif _train_dataset == 'cifar10':
        _img_width = cfg_para_yml.img_width_cifar10
        _img_height = cfg_para_yml.img_height_cifar10
        _num_channels = cfg_para_yml.num_channels_rgb
        _num_classes = cfg_para_yml.num_class_cifar10
        _cifar10_path = cfg_path_yml.cifar_10_path
        train_data, train_label, test_data, test_label = load_cifar10_keras()
        use_raw_image = False

    ########################################
    # Processes for training
    ########################################

    os_thread_num = mp.cpu_count()
    _available_gpu_num = cfg_para_yml.available_gpu_num
    training_gpu_queue, training_cpu_queue = generate_job_queue()

    proc_gpu_list = list()

    for gn in range(_available_gpu_num):
        assign_gpu = '/gpu:' + str(gn)
        device_proc_gpu = mp.Process(target=consumer_gpu, args=(training_gpu_queue, assign_gpu))
        proc_gpu_list.append(device_proc_gpu)

    for device_proc_gpu in proc_gpu_list:
        device_proc_gpu.start()

    device_proc_cpu = mp.Process(target=consumer_cpu, args=(training_cpu_queue, '/cpu:0'))
    device_proc_cpu.start()

    for device_proc_gpu in proc_gpu_list:
        device_proc_gpu.start()
