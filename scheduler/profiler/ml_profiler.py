import numpy as np
import json
import argparse
import itertools
from timeit import default_timer as timer
import os

import tensorflow as tf

from tensorflow_cifar.models.alexnet import AlexNet
from tensorflow_cifar.models.densenet import DenseNet
from tensorflow_cifar.models.efficientnet import EfficientNet
from tensorflow_cifar.models.inception import Inception
from tensorflow_cifar.models.lenet import LeNet
from tensorflow_cifar.models.mobilenet import MobileNet
from tensorflow_cifar.models.mobilenet_v2 import MobileNetV2
from tensorflow_cifar.models.resnet import ResNet
from tensorflow_cifar.models.resnext import ResNeXt
from tensorflow_cifar.models.shufflenet import ShuffleNet
from tensorflow_cifar.models.shufflenet_v2 import ShuffleNetV2
from tensorflow_cifar.models.squeezenet import SqueezeNet
from tensorflow_cifar.models.vgg import VGG
from tensorflow_cifar.models.xception import Xception
from tensorflow_cifar.models.zfnet import ZFNet

from tensorflow_cifar.tools.dataset_loader import load_cifar10_keras
from tensorflow_cifar.tools.model_tools import train_model
from tensorflow_cifar.tools.model_tools import evaluate_model


if __name__ == "__main__":

    ###################################
    # get all parameters
    ###################################

    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--profile', action='store', type=str, choices=['accuracy', 'steptime'],
                        help='profile model accuracy or steptime')
    parser.add_argument('-m', '--model', action='store', type=str, help='indicate training model')
    parser.add_argument('-l', '--layer', action='store', type=int,
                        help='indicate the layer for some models like resnet, densenet, vgg')
    parser.add_argument('-g', '--group', action='store', type=int,
                        help='indicate the conv group for shufflenet')
    parser.add_argument('-x', '--complex', action='store', type=float,
                        help='indicate the model complex for shufflenetv2')
    parser.add_argument('-c', '--card', action='store', type=int, help='indicate the cardinality for resnext')

    parser.add_argument('-b', '--batch', action='store', type=int, help='indicate the batch size for training.')
    parser.add_argument('-r', '--learn', action='store', type=float, help='indicate the learning rate for training.')
    parser.add_argument('-o', '--opt', action='store', type=str, help='indicate the optimizer for training.')
    parser.add_argument('-e', '--epoch', action='store', type=int, help='indicate the training epoch.')

    args = parser.parse_args()

    profile_mode = args.profile
    model_name = args.model
    model_layer = args.layer
    model_group = args.group
    model_complex = args.complex
    model_card = args.card

    batch_size = args.batch
    learn_rate = args.learn
    optimizer = args.opt
    epoch = args.epoch

    num_output_classes = 10

    ###################################
    # prepare the model
    ###################################

    if model_name == 'alexnet':
        model = AlexNet(num_output_classes)
    elif model_name == 'densenet':
        model = DenseNet(model_layer, num_output_classes)
    elif model_name == 'efficientnet':
        model = EfficientNet(num_output_classes)
    elif model_name == 'inception':
        model = Inception(num_output_classes)
    elif model_name == 'lenet':
        model = LeNet(num_output_classes)
    elif model_name == 'mobilenet':
        model = MobileNet(num_output_classes)
    elif model_name == 'mobilenetv2':
        model = MobileNetV2(num_output_classes)
    elif model_name == 'resnet':
        model = ResNet(model_layer, num_output_classes)
    elif model_name == 'resnext':
        model = ResNeXt(model_card, num_output_classes)
    elif model_name == 'shufflenet':
        model = ShuffleNet(model_group, num_output_classes)
    elif model_name == 'shufflenetv2':
        model = ShuffleNetV2(model_complex, num_output_classes)
    elif model_name == 'squeezenet':
        model = SqueezeNet(num_output_classes)
    elif model_name == 'vgg':
        model = VGG(model_layer, num_output_classes)
    elif model_name == 'xception':
        model = Xception(num_output_classes)
    elif model_name == 'zfnet':
        model = ZFNet(num_output_classes)
    else:
        raise ValueError('model is unsupported')

    feature_ph = tf.placeholder(tf.float32, [None, 32, 32, 3])
    label_ph = tf.placeholder(tf.int32, [None, 10])
    logit = model.build(feature_ph)

    train_op = train_model(logit, label_ph, optimizer, learn_rate)
    eval_op = evaluate_model(logit, label_ph)

    ###########################################
    # count overall trainable parameters
    ###########################################

    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters

    ###############################
    # load dataset
    ###############################

    train_feature, train_label, eval_feature, eval_label = load_cifar10_keras()

    ###################################
    # profile the model
    ###################################

    print('Start Profiling {}'.format(model_name))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        num_feature = train_label.shape[0]
        num_batch = num_feature // batch_size
        rest_feature = num_feature - batch_size * num_batch
        acc_record_list = list()
        time_record_list = list()

        # train the model
        for e in range(epoch):

            start_time = timer()

            # shuffle the training data
            shf_indices = np.arange(num_feature)
            np.random.shuffle(shf_indices)
            train_feature = train_feature[shf_indices]
            train_label = train_label[shf_indices]

            for i in range(num_batch):
                print('epoch %d / %d, step %d / %d' % (e + 1, epoch, i + 1, num_batch))

                batch_offset = i * batch_size
                batch_end = (i + 1) * batch_size
                train_feature_batch = train_feature[batch_offset:batch_end]
                train_label_batch = train_label[batch_offset:batch_end]
                sess.run(train_op, feed_dict={feature_ph: train_feature_batch, label_ph: train_label_batch})

            if rest_feature != 0:
                print('the rest train feature: {}, train them now'.format(rest_feature))
                rest_feature_batch = train_feature[-rest_feature:]
                rest_label_batch = train_label[-rest_feature:]
                sess.run(train_op, feed_dict={feature_ph: rest_feature_batch, label_ph: rest_label_batch})
            else:
                print('no train feature left for this epoch')

            end_time = timer()
            step_time = end_time - start_time
            time_record_list.append(step_time)

            print('start evaluation phrase')
            acc_sum = 0
            eval_batch_size = 50
            num_batch_eval = eval_label.shape[0] // eval_batch_size
            for i in range(num_batch_eval):
                print('evaluation step %d / %d' % (i + 1, num_batch_eval))
                batch_offset = i * eval_batch_size
                batch_end = (i + 1) * eval_batch_size
                eval_feature_batch = eval_feature[batch_offset:batch_end]
                eval_label_batch = eval_label[batch_offset:batch_end]
                acc_batch = sess.run(eval_op, feed_dict={feature_ph: eval_feature_batch, label_ph: eval_label_batch})
                acc_sum += acc_batch

            acc_avg = acc_sum / num_batch_eval
            print('evaluation accuracy:{}'.format(acc_avg))
            acc_record_list.append(acc_avg)

    steptime_avg = sum(time_record_list) / len(time_record_list)

    json_acc_path = '/home/ruiliu/Development/ml-estimator/mlbase/model_acc.json'
    json_time_path = '/home/ruiliu/Development/ml-estimator/mlbase/model_time_simulate.json'

    # json_acc_path = '/home/ruiliu/Development/ml-estimator/mlbase/model_acc.json'
    # json_time_path = '/home/ruiliu/Development/ml-estimator/mlbase/model_time.json'

    if profile_mode == 'accuracy':
        if os.path.exists(json_acc_path):
            with open(json_acc_path) as f:
                model_json_list = json.load(f)
        else:
            model_json_list = list()

        # create a dict for the conf
        model_perf_dict = dict()

        model_perf_dict['model_name'] = model_name
        model_perf_dict['num_parameters'] = total_parameters
        # model_perf_dict['num_parameters'] = np.random.randint(low=1000000, high=5000000)
        model_perf_dict['batch_size'] = batch_size
        model_perf_dict['opt'] = optimizer
        model_perf_dict['learn_rate'] = learn_rate

        model_perf_dict['training_data'] = 'cifar'
        model_perf_dict['classes'] = num_output_classes

        model_perf_dict['accuracy'] = acc_record_list
        # model_perf_dict['accuracy'] = list(list(np.random.uniform(0, 1, 20)))

        model_json_list.append(model_perf_dict)

        with open(json_acc_path, 'w+') as f:
            json.dump(model_json_list, f)

    else:
        if os.path.exists(json_time_path):
            with open(json_time_path) as f:
                model_json_list = json.load(f)
        else:
            model_json_list = list()

        # create a dict for the conf
        model_perf_dict = dict()

        model_perf_dict['model_name'] = model_name
        model_perf_dict['num_parameters'] = total_parameters
        model_perf_dict['batch_size'] = batch_size
        model_perf_dict['input_chn'] = train_feature.shape[-1]
        model_perf_dict['input_size'] = train_feature.shape[1] * train_feature.shape[2]
        model_perf_dict['opt'] = optimizer
        model_perf_dict['learn_rate'] = learn_rate

        model_perf_dict['training_data'] = 'cifar'
        model_perf_dict['classes'] = num_output_classes

        model_perf_dict['steptime'] = steptime_avg
        # model_perf_dict['steptime'] = np.random.randint(low=1, high=100)

        model_json_list.append(model_perf_dict)

        with open(json_time_path, 'w+') as f:
            json.dump(model_json_list, f)
