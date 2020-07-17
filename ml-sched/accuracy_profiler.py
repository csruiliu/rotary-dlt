import numpy as np
import tensorflow as tf
import argparse

import config_path as cfg_path_yml
import config_parameter as cfg_para_yml
from model_importer import ModelImporter
from utils_img_func import *


def build_model():
    model_name_abbr = np.random.choice(rand_seed, 1, replace=False).tolist()
    dm = ModelImporter(train_model, str(model_name_abbr.pop()), train_conv_layer, img_height, img_width, num_channels,
                       num_classes, train_batchsize, train_opt, train_learn_rate, train_activation, False)
    model_entity = dm.get_model_entity()
    model_logit = model_entity.build(features)
    conv_layer_num, pool_layer_num, total_layer_num = model_entity.get_layer_info()

    model_name_output = '{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}'.format(input_data_size, input_data_channel, output_class,
                                                                  train_batchsize, conv_layer_num, pool_layer_num,
                                                                  total_layer_num, train_learn_rate, train_opt,
                                                                  train_activation, train_epoch)
    train_step = model_entity.train(model_logit, labels)
    eval_step = model_entity.evaluate(model_logit, labels)
    return train_step, eval_step, model_name_output


def run_train_model(trainOp):
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        num_batch = Y_data.shape[0] // train_batchsize

        for e in range(train_epoch):
            for i in range(num_batch):
                print('epoch {} / {}, step {} / {}'.format(e+1, train_epoch, i+1, num_batch))
                batch_offset = i * train_batchsize
                batch_end = (i+1) * train_batchsize
                X_mini_batch_feed = X_data[batch_offset:batch_end, :, :, :]
                Y_mini_batch_feed = Y_data[batch_offset:batch_end, :]
                sess.run(trainOp, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})

    print('Finish training model')


def run_eval_model(evalOp, model_info):
    print('start evaluating model')
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        acc_arg = sess.run(evalOp, feed_dict={features: X_data_eval, labels: Y_data_eval})

    print("{{\"model_name\": \"{}\", \"model_accuracy\": {}}}".format(model_info, acc_arg))


if __name__ == '__main__':

    #########################################################################
    # Constant parameters
    #########################################################################

    rand_seed = 10000

    #########################################################################
    # Parameters read from command
    #########################################################################

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', required=True, action='store', type=str,
                        help='model type [resnet, mobilenet, mlp, densenet, scn]')

    parser.add_argument('-b', '--batchsize', required=True, action='store', type=int,
                        help='batch size, for example: 32, 50, 100]')

    parser.add_argument('-d', '--dataset', required=True, action='store', type=str,
                        help='training set [imagenet, cifar10]')

    parser.add_argument('-e', '--epoch', required=True, action='store', type=int,
                        help='training epoch, for example, 1, 5, 10')

    parser.add_argument('-o', '--opt', required=True, action='store', type=str,
                        help='training opt [Adam, SGD, Adagrad, Momentum]')

    parser.add_argument('-l', '--conv_layer', required=True, action='store', type=int,
                        help='number of training conv layer, for example, 1, 2, 3')

    parser.add_argument('-r', '--learn_rate', required=True, action='store', type=float,
                        help='learning rate, for example, 0.01, 0.001, 0.0001')

    parser.add_argument('-a', '--activation', required=True, action='store', type=str,
                        help='activation, for example, relu, sigmoid, tanh, leaky_relu')

    args = parser.parse_args()

    train_data = args.dataset
    train_model = args.model
    train_batchsize = args.batchsize
    train_epoch = args.epoch
    train_opt = args.opt
    train_conv_layer = args.conv_layer
    train_learn_rate = args.learn_rate
    train_activation = args.activation

    train_op = None
    eval_op = None
    img_width = None
    img_height = None
    num_channels = None
    num_classes = None

    if train_data == 'imagenet':
        print('train the model on imagenet')
        img_width = cfg_para_yml.img_width_imagenet
        img_height = cfg_para_yml.img_height_imagenet
        num_channels = cfg_para_yml.num_channels_rgb
        num_classes = cfg_para_yml.num_class_imagenet
        input_data_size = 224
        input_data_channel = 3
        output_class = 1000

        train_image_path = cfg_path_yml.imagenet_t10k_bin_path
        train_label_path = cfg_path_yml.imagenet_t10k_label_path
        test_image_path = cfg_path_yml.imagenet_t1k_bin_path
        test_label_path = cfg_path_yml.imagenet_t1k_label_path

        X_data = load_imagenet_bin(train_image_path, num_channels, img_width, img_height)
        Y_data = load_imagenet_labels_onehot(train_label_path, num_classes)
        X_data_eval = load_imagenet_bin(test_image_path, num_channels, img_width, img_height)
        Y_data_eval = load_imagenet_labels_onehot(test_label_path, num_classes)

    elif train_data == 'cifar10':
        print('train the model on cifar10')
        img_width = cfg_para_yml.img_width_cifar10
        img_height = cfg_para_yml.img_height_cifar10
        num_channels = cfg_para_yml.num_channels_rgb
        num_classes = cfg_para_yml.num_class_cifar10
        input_data_size = 32
        input_data_channel = 3
        output_class = 10

        cifar10_path = cfg_path_yml.cifar_10_path
        X_data, Y_data = load_cifar_train(cifar10_path, rand_seed)
        X_data_eval, Y_data_eval = load_cifar_test(cifar_10_path, seed)

    elif train_data == 'mnist':
        print('train the model on mnist')
        img_width = cfg_para_yml.img_width_mnist
        img_height = cfg_para_yml.img_height_mnist
        num_channels = cfg_para_yml.num_channels_bw
        num_classes = cfg_para_yml.num_class_mnist
        input_data_size = 28
        input_data_channel = 1
        output_class = 10

        mnist_train_img_path = cfg_path_yml.mnist_train_img_path
        mnist_train_label_path = cfg_path_yml.mnist_train_label_path
        mnist_t10k_img_path = cfg_path_yml.mnist_t10k_img_path
        mnist_t10k_label_path = cfg_path_yml.mnist_t10k_label_path

        X_data = load_mnist_image(mnist_train_img_path, rand_seed)
        Y_data = load_mnist_label_onehot(mnist_train_label_path, rand_seed)
        X_data_eval = load_mnist_image(mnist_t10k_img_path, rand_seed)
        Y_data_eval = load_mnist_label_onehot(mnist_t10k_label_path, rand_seed)

    features = tf.placeholder(tf.float32, [None, img_width, img_height, num_channels])
    labels = tf.placeholder(tf.int64, [None, num_classes])
    train_op, eval_op, train_model_name = build_model()

    run_train_model(train_op)
    run_eval_model(eval_op, train_model_name)