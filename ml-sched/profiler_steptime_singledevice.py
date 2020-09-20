import tensorflow as tf
import argparse
import os
from timeit import default_timer as timer

import config_path as cfg_path_yml
import config_parameter as cfg_para_yml
from model_importer import ModelImporter
from utils_img_func import *


def build_train_model():
    with tf.device(train_device):
        features = tf.placeholder(tf.float32, [None, img_width, img_height, num_channels])
        labels = tf.placeholder(tf.int64, [None, num_classes])

        model_name_abbr = np.random.choice(rand_seed, train_model_num, replace=False).tolist()

        train_op_list = list()
        total_conv_layer = 0
        total_pool_layer = 0
        total_residual_layer = 0
        for i in range(train_model_num):
            dm = ModelImporter(train_model_type, str(model_name_abbr.pop()), train_model_layer, img_height,
                               img_width, num_channels, num_classes, train_batchsize, train_opt,
                               train_learn_rate, train_activation, False)
            model_entity = dm.get_model_entity()
            model_logit = model_entity.build(input_features=features, is_training=True)
            train_step = model_entity.train(model_logit, labels)
            train_op_list.append(train_step)

            conv_layer_num, pool_layer_num, residual_layer_num = model_entity.get_layer_info()
            total_conv_layer += conv_layer_num
            total_pool_layer += pool_layer_num
            total_residual_layer += residual_layer_num

        data_cost_info = '{0}-{1}-{2}-{3}'.format(img_width, num_channels, num_classes, train_batchsize*train_model_num)
        model_layer_info = '{0}-{1}-{2}'.format(total_conv_layer, total_pool_layer, total_residual_layer)
        # FP32 (float) performance TFLOPS
        gpu_tflops_info = '16.31'

        #########################################################################
        # Traing the model
        #########################################################################

        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.gpu_options.allow_growth = True

        if train_dataset == 'imagenet':
            image_list = sorted(os.listdir(imagenet_train_img_path))

        step_time = 0
        step_count = 0
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            num_batch = train_label.shape[0] // train_batchsize
            for e in range(train_epoch):
                for i in range(num_batch):

                    if i != 0:
                        start_time = timer()

                    print('epoch {} / {}, step {} / {}'.format(e+1, train_epoch, i+1, num_batch))
                    batch_offset = i * train_batchsize
                    batch_end = (i+1) * train_batchsize

                    if train_dataset == 'imagenet':
                        batch_list = image_list[batch_offset:batch_end]
                        X_mini_batch_feed = load_imagenet_raw(imagenet_train_label_path, batch_list, img_height, img_width)
                    else:
                        X_mini_batch_feed = train_feature[batch_offset:batch_end]

                    Y_mini_batch_feed = train_label[batch_offset:batch_end]

                    sess.run(train_op_list, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})

                    if i != 0:
                        end_time = timer()
                        dur_time = end_time - start_time
                        print("step time:", dur_time)
                        step_time += dur_time
                        step_count += 1

    avg_step_time = step_time / step_count * 1000

    print("{{\"data_cost\": \"{0}\", \"model_cost\": \"{1}\", \"tflops_cost\": \"{2}\", \"model_step_time\": {3}}}".format(data_cost_info, model_layer_info, gpu_tflops_info, avg_step_time))


if __name__ == '__main__':

    #########################################################################
    # Constant parameters
    #########################################################################

    rand_seed = 10000

    #########################################################################
    # Parameters read from command
    #########################################################################

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model_type', required=True, action='store',
                        choices=['resnet', 'mobilenet', 'densenet', 'mlp', 'scn'],
                        help='model type [resnet, mobilenet, mlp, densenet, scn]')

    parser.add_argument('-n', '--model_num', required=True, action='store', type=int,
                        help='the number of training model on the device')

    parser.add_argument('-e', '--epoch', required=True, action='store', type=int,
                        help='training epoch, for example, 1, 5, 10')

    parser.add_argument('-b', '--batch_size', required=True, action='store', type=int,
                        help='batch size, for example: 32, 50, 100]')

    parser.add_argument('-o', '--opt', required=True, action='store', choices=['SGD', 'Adam', 'Adagrad', 'Momentum'],
                        help='training opt [Adam, SGD, Adagrad, Momentum]')

    parser.add_argument('-r', '--learn_rate', required=True, action='store', type=float,
                        help='learning rate, for example, 0.01, 0.001, 0.0001')

    parser.add_argument('-a', '--activation', required=True, action='store',
                        choices=['relu', 'sigmoid', 'tanh', 'leaky_relu'],
                        help='activation, for example, relu, sigmoid, tanh, leaky_relu')

    parser.add_argument('-t', '--train_dataset', required=True, action='store', choices=['imagenet', 'cifar10'],
                        help='training set [imagenet, cifar10]')

    parser.add_argument('-l', '--layer', action='store', type=int,
                        help='the number of layer decides a model, for example, the layer is 50 for resnet-50')

    parser.add_argument('-d', '--device', action='store', type=str, default='gpu:0',
                        choices=['cpu:0', 'gpu:0', 'gpu:1'],
                        help='select a device to run device')

    args = parser.parse_args()


    train_model_type = args.model_type
    train_model_num = args.model_num
    train_batchsize = args.batch_size
    train_epoch = args.epoch
    train_opt = args.opt
    train_learn_rate = args.learn_rate
    train_activation = args.activation

    train_device = args.device
    train_pack = args.pack
    print(train_pack)
    train_model_layer = args.layer
    train_dataset = args.train_dataset

    train_op = None
    eval_op = None
    img_width = None
    img_height = None
    num_channels = None
    num_classes = None

    if train_dataset == 'imagenet':
        print('train the model on imagenet')
        img_width = cfg_para_yml.img_width_imagenet
        img_height = cfg_para_yml.img_height_imagenet
        num_channels = cfg_para_yml.num_channels_rgb
        num_classes = cfg_para_yml.num_class_imagenet

        imagenet_train_img_path = cfg_path_yml.imagenet_t50k_img_raw_path
        imagenet_train_label_path = cfg_path_yml.imagenet_t50k_label_path

        train_label = load_imagenet_labels_onehot(imagenet_train_label_path, num_classes)

    elif train_dataset == 'cifar10':
        print('train the model on cifar10')
        img_width = cfg_para_yml.img_width_cifar10
        img_height = cfg_para_yml.img_height_cifar10
        num_channels = cfg_para_yml.num_channels_rgb
        num_classes = cfg_para_yml.num_class_cifar10

        cifar10_path = cfg_path_yml.cifar_10_path
        train_feature, train_label, test_feature, test_label = load_cifar10_keras()
        # X_data, Y_data = load_cifar10_train(cifar10_path)
        # X_data_eval, Y_data_eval = load_cifar10_test(cifar10_path)

    elif train_dataset == 'mnist':
        print('train the model on mnist')
        img_width = cfg_para_yml.img_width_mnist
        img_height = cfg_para_yml.img_height_mnist
        num_channels = cfg_para_yml.num_channels_bw
        num_classes = cfg_para_yml.num_class_mnist

        mnist_train_img_path = cfg_path_yml.mnist_train_img_path
        mnist_train_label_path = cfg_path_yml.mnist_train_label_path

        train_feature = load_mnist_image(mnist_train_img_path)
        train_label = load_mnist_label_onehot(mnist_train_label_path)

    build_train_model()
