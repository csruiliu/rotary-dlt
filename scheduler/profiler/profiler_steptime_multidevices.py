import tensorflow as tf
import argparse
import multiprocessing as mp
import numpy as np
from timeit import default_timer as timer
import os

import relish.config.config_parameter as cfg_para
from relish.common.model_importer import ModelImporter
from relish.common.dataset_loader import load_dataset_para, load_train_dataset
from relish.tools.img_tool import load_imagenet_raw


def generate_job_queue():
    cpu_job_queue = mp.Queue()
    gpu_job_queue = mp.Queue()

    workload_num = cpu_model_num + gpu_model_num
    _rand_seed = 10000
    np.random.seed(_rand_seed)
    model_name_abbr = np.random.choice(_rand_seed, workload_num, replace=False).tolist()

    for _ in range(gpu_model_num):
        if not gpu_job_queue.full():
            gpu_job_queue.put([gpu_model_type,
                               model_name_abbr.pop(),
                               gpu_model_layer,
                               gpu_batch_size,
                               gpu_optimizer,
                               gpu_learn_rate,
                               gpu_activation])

    for _ in range(cpu_model_num):
        if not cpu_job_queue.full():
            cpu_job_queue.put([cpu_model_type,
                               model_name_abbr.pop(),
                               cpu_model_layer,
                               cpu_batch_size,
                               cpu_optimizer,
                               cpu_learn_rate,
                               cpu_activation])

    return gpu_job_queue, cpu_job_queue


def run_single_job(model_type,
                   model_instance,
                   layer_num,
                   batch_size,
                   optimizer,
                   learning_rate,
                   activation,
                   assign_device,
                   proc_idx=0):
    with tf.device(assign_device):
        feature_ph = tf.placeholder(tf.float32, [None, img_w, img_h, num_chn])
        label_ph = tf.placeholder(tf.int64, [None, num_cls])

        train_model = ModelImporter(model_type, str(model_instance),
                                    layer_num, img_h, img_w, num_chn,
                                    num_cls, batch_size, optimizer,
                                    learning_rate, activation, False)

        model_entity = train_model.get_model_entity()
        model_logit = model_entity.build(feature_ph, is_training=True)
        train_ops = model_entity.train(model_logit, label_ph)

        num_conv_layer, num_pool_layer, num_residual_layer = model_entity.get_layer_info()

        if train_dataset == 'imagenet':
            image_list = sorted(os.listdir(train_feature_input))

        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        tf_config.allow_soft_placement = True

        print('{} job start...'.format(assign_device))

        step_time = 0
        step_count = 0

        with tf.Session(config=tf_config) as sess:
            sess.run(tf.global_variables_initializer())
            num_batch = train_label_input.shape[0] // batch_size
            for i in range(num_batch):
                if assign_device.startswith('/cpu'):
                    print('**CPU JOB**: Proc-{}, {}-{}-{} on cpu [{}]: step {} / {}'
                          .format(proc_idx, model_type, batch_size, model_instance, timer(), i+1, num_batch))
                elif assign_device.startswith('/gpu'):
                    print('**GPU JOB**: {}-{}-{} on cpu [{}]: step {} / {}'
                          .format(model_type, batch_size, model_instance, timer(), i+1, num_batch))
                if i != 0:
                    start_time = timer()

                batch_offset = i * batch_size
                batch_end = (i + 1) * batch_size
                if train_dataset == 'imagenet':
                    batch_list = image_list[batch_offset:batch_end]
                    train_feature_batch = load_imagenet_raw(train_feature_input,
                                                          batch_list,
                                                          img_h, img_w)
                else:
                    train_feature_batch = train_feature_input[batch_offset:batch_end]
                train_label_batch = train_label_input[batch_offset:batch_end]
                sess.run(train_ops, feed_dict={feature_ph: train_feature_batch, label_ph: train_label_batch})

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
        p = mp.Process(target=run_single_job, args=(gpu_job[0],
                                                    gpu_job[1],
                                                    gpu_job[2],
                                                    gpu_job[3],
                                                    gpu_job[4],
                                                    gpu_job[5],
                                                    gpu_job[6],
                                                    assign_device))
        p.start()
        p.join()


def consumer_cpu(queue, assign_device):
    for proc_idx in range(os_thread_num):
        if not queue.empty():
            cpu_job = queue.get()
            p = mp.Process(target=run_single_job, args=(cpu_job[0],
                                                        cpu_job[1],
                                                        cpu_job[2],
                                                        cpu_job[3],
                                                        cpu_job[4],
                                                        cpu_job[5],
                                                        cpu_job[6],
                                                        assign_device,
                                                        proc_idx))
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

    cpu_model_type = args.cpu_model
    cpu_model_layer = args.cpu_model_layer
    cpu_model_num = args.cpu_model_num
    cpu_batch_size = args.cpu_batch_size
    cpu_learn_rate = args.cpu_learn_rate
    cpu_activation = args.cpu_activation
    cpu_optimizer = args.cpu_optimizer

    gpu_model_type = args.gpu_model
    gpu_model_layer = args.gpu_model_layer
    gpu_model_num = args.gpu_model_num
    gpu_batch_size = args.gpu_batch_size
    gpu_learn_rate = args.gpu_learn_rate
    gpu_activation = args.gpu_activation
    gpu_optimizer = args.gpu_optimizer

    train_dataset = args.train_dataset

    img_w, img_h, num_chn, num_cls = load_dataset_para(train_dataset)
    train_feature_input, train_label_input = load_train_dataset(train_dataset)

    ########################################
    # Processes for training
    ########################################

    os_thread_num = mp.cpu_count()
    available_gpu_num = cfg_para.sch_gpu_num
    training_gpu_queue, training_cpu_queue = generate_job_queue()

    proc_gpu_list = list()

    for gn in range(available_gpu_num):
        assign_gpu = '/gpu:' + str(gn)
        device_proc_gpu = mp.Process(target=consumer_gpu, args=(training_gpu_queue, assign_gpu))
        proc_gpu_list.append(device_proc_gpu)

    for device_proc_gpu in proc_gpu_list:
        device_proc_gpu.start()

    device_proc_cpu = mp.Process(target=consumer_cpu, args=(training_cpu_queue, '/cpu:0'))
    device_proc_cpu.start()

    for device_proc_gpu in proc_gpu_list:
        device_proc_gpu.start()
