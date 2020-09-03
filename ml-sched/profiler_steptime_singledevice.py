import tensorflow as tf
import argparse
import os
from timeit import default_timer as timer

import config_path as cfg_path_yml
import config_parameter as cfg_para_yml
from model_importer import ModelImporter
from utils_img_func import *


def build_model():
    with tf.device(train_device):
        model_name_abbr = np.random.choice(rand_seed, 1, replace=False).tolist()
        dm = ModelImporter(train_model_type, str(model_name_abbr.pop()), train_layer, img_height, img_width, num_channels,
                           num_classes, train_batchsize, train_opt, train_learn_rate, train_activation, False)
        model_entity = dm.get_model_entity()
        model_logit = model_entity.build(input_features=features, is_training=True)
        conv_layer_num, pool_layer_num, residual_layer_num = model_entity.get_layer_info()

        data_cost_info = '{0}-{1}-{2}-{3}'.format(input_data_size, num_channels, num_classes, train_batchsize)
        model_layer_info = '{0}-{1}-{2}'.format(conv_layer_num, pool_layer_num, residual_layer_num)
        # FP32 (float) performance TFLOPS
        gpu_tflops_info = '16.31'

        train_step = model_entity.train(model_logit, labels)

    return train_step, data_cost_info, model_layer_info, gpu_tflops_info


def train_model(trainOp, data_info, model_info, tflops_info):
    with tf.device(train_device):
        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.gpu_options.allow_growth = True

        if train_data == 'imagenet':
            image_list = sorted(os.listdir(imagenet_train_img_path))

        step_time = 0
        step_count = 0
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            num_batch = Y_data.shape[0] // train_batchsize

            for e in range(train_epoch):
                for i in range(num_batch):

                    if i != 0:
                        start_time = timer()

                    print('epoch {} / {}, step {} / {}'.format(e+1, train_epoch, i+1, num_batch))
                    batch_offset = i * train_batchsize
                    batch_end = (i+1) * train_batchsize

                    if train_data == 'imagenet':
                        batch_list = image_list[batch_offset:batch_end]
                        X_mini_batch_feed = load_imagenet_raw(imagenet_test_img_raw_path, batch_list, img_height, img_width)
                    else:
                        X_mini_batch_feed = X_data[batch_offset:batch_end]

                    Y_mini_batch_feed = Y_data[batch_offset:batch_end]
                    sess.run(trainOp, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})

                    if i != 0:
                        end_time = timer()
                        dur_time = end_time - start_time
                        print("step time:", dur_time)
                        step_time += dur_time
                        step_count += 1

    avg_step_time = step_time / step_count * 1000

    print("{{\"data_cost\": \"{0}\", \"model_cost\": \"{1}\", \"tflops_cost\": \"{2}\", \"model_step_time\": {3}}}".format(data_info, model_info, tflops_info, avg_step_time))


if __name__ == '__main__':

    #########################################################################
    # Constant parameters
    #########################################################################

    rand_seed = 10000

    #########################################################################
    # Parameters read from command
    #########################################################################

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', required=True, action='store',
                        choices=['resnet', 'mobilenet', 'densenet', 'mlp', 'scn'],
                        help='model type [resnet, mobilenet, mlp, densenet, scn]')

    parser.add_argument('-b', '--batchsize', required=True, action='store', type=int,
                        help='batch size, for example: 32, 50, 100]')

    parser.add_argument('-t', '--train_set', required=True, action='store', choices=['imagenet', 'cifar10'],
                        help='training set [imagenet, cifar10]')

    parser.add_argument('-e', '--epoch', required=True, action='store', type=int,
                        help='training epoch, for example, 1, 5, 10')

    parser.add_argument('-o', '--opt', required=True, action='store', choices=['SGD', 'Adam', 'Adagrad', 'Momentum'],
                        help='training opt [Adam, SGD, Adagrad, Momentum]')

    parser.add_argument('-r', '--learn_rate', required=True, action='store', type=float,
                        help='learning rate, for example, 0.01, 0.001, 0.0001')

    parser.add_argument('-a', '--activation', required=True, action='store',
                        choices=['relu', 'sigmoid', 'tanh', 'leaky_relu'],
                        help='activation, for example, relu, sigmoid, tanh, leaky_relu')

    parser.add_argument('-l', '--layer', action='store', type=int,
                        help='the number of layer decides a model, for example, the layer is 50 for resnet-50')

    parser.add_argument('-d', '--device', action='store', type=str, default='/GPU:0', choices=['/GPU:0', '/GPU:1'],
                        help='select a device to run device')

    args = parser.parse_args()

    #if args.model not in ['mlp', 'scn'] and args.conv_layer:
    #    train_conv_layer = args.conv_layer
    #else:
    #    train_conv_layer = 1

    train_data = args.train_set
    train_model_type = args.model
    train_layer = args.layer
    train_batchsize = args.batchsize
    train_epoch = args.epoch
    train_opt = args.opt
    train_learn_rate = args.learn_rate
    train_activation = args.activation
    train_device = args.device

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

        imagenet_train_img_path = cfg_path_yml.imagenet_t50k_img_raw_path
        imagenet_train_label_path = cfg_path_yml.imagenet_t50k_img_label_path

        imagenet_test_img_raw_path = cfg_path_yml.imagenet_t1k_img_raw_path
        imagenet_test_img_bin_path = cfg_path_yml.imagenet_t1k_img_bin_path
        imagenet_test_label_path = cfg_path_yml.imagenet_t1k_label_path

        Y_data = load_imagenet_labels_onehot(imagenet_train_label_path, num_classes)
        X_data_eval = load_imagenet_bin(imagenet_test_img_bin_path, num_channels, img_width, img_height)
        Y_data_eval = load_imagenet_labels_onehot(imagenet_test_label_path, num_classes)

    elif train_data == 'cifar10':
        print('train the model on cifar10')
        img_width = cfg_para_yml.img_width_cifar10
        img_height = cfg_para_yml.img_height_cifar10
        num_channels = cfg_para_yml.num_channels_rgb
        num_classes = cfg_para_yml.num_class_cifar10
        input_data_size = 32

        cifar10_path = cfg_path_yml.cifar_10_path
        X_data, Y_data, X_data_eval, Y_data_eval = load_cifar10_keras()
        # X_data, Y_data = load_cifar10_train(cifar10_path)
        # X_data_eval, Y_data_eval = load_cifar10_test(cifar10_path)

    elif train_data == 'mnist':
        print('train the model on mnist')
        img_width = cfg_para_yml.img_width_mnist
        img_height = cfg_para_yml.img_height_mnist
        num_channels = cfg_para_yml.num_channels_bw
        num_classes = cfg_para_yml.num_class_mnist
        input_data_size = 28

        mnist_train_img_path = cfg_path_yml.mnist_train_img_path
        mnist_train_label_path = cfg_path_yml.mnist_train_label_path
        mnist_test_img_path = cfg_path_yml.mnist_test_10k_img_path
        mnist_test_label_path = cfg_path_yml.mnist_test_10k_label_path

        X_data = load_mnist_image(mnist_train_img_path, rand_seed)
        Y_data = load_mnist_label_onehot(mnist_train_label_path, rand_seed)
        X_data_eval = load_mnist_image(mnist_test_img_path, rand_seed)
        Y_data_eval = load_mnist_label_onehot(mnist_test_label_path, rand_seed)

    features = tf.placeholder(tf.float32, [None, img_width, img_height, num_channels])
    labels = tf.placeholder(tf.int64, [None, num_classes])
    train_op, data_cost_info, model_layer_info, gpu_tflops_info = build_model()

    train_model(train_op, data_cost_info, model_layer_info, gpu_tflops_info)
