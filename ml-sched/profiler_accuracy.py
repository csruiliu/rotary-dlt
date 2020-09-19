import tensorflow as tf
import argparse
import os

import config_path as cfg_path_yml
import config_parameter as cfg_para_yml
from model_importer import ModelImporter
from utils_img_func import *


def build_model():
    with tf.device(train_device):
        model_name_abbr = np.random.choice(rand_seed, 1, replace=False).tolist()
        dm = ModelImporter(train_model, str(model_name_abbr.pop()), train_layer, img_height, img_width, num_channels,
                           num_classes, train_batchsize, train_opt, train_learn_rate, train_activation, False)
        model_entity = dm.get_model_entity()
        model_logit = model_entity.build(input_features=features, is_training=True)
        conv_layer_num, pool_layer_num, residual_layer_num = model_entity.get_layer_info()

        model_name_output = '{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}'.format(input_data_size, num_channels, num_classes,
                                                                      train_batchsize, conv_layer_num, pool_layer_num,
                                                                      residual_layer_num, train_learn_rate, train_opt,
                                                                      train_activation, train_epoch)
        train_step = model_entity.train(model_logit, labels)
        eval_step = model_entity.evaluate(model_logit, labels)

    return train_step, eval_step, model_name_output


def train_eval_model(trainOp, evalOp, model_info):
    with tf.device(train_device):
        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.gpu_options.allow_growth = True

        if train_data == 'imagenet':
            image_list_train = sorted(os.listdir(imagenet_train_img_path))
            image_list_eval = sorted(os.listdir(imagenet_test_img_path))

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            num_batch = Y_data_train.shape[0] // train_batchsize

            for e in range(train_epoch):
                for i in range(num_batch):
                    print('epoch {} / {}, step {} / {}'.format(e+1, train_epoch, i+1, num_batch))
                    batch_offset = i * train_batchsize
                    batch_end = (i+1) * train_batchsize

                    if train_data == 'imagenet':
                        batch_list = image_list_train[batch_offset:batch_end]
                        X_mini_batch_feed = load_imagenet_raw(imagenet_train_img_path, batch_list, img_height, img_width)
                    else:
                        X_mini_batch_feed = X_data[batch_offset:batch_end]

                    Y_mini_batch_feed = Y_data_train[batch_offset:batch_end]
                    sess.run(trainOp, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})

            if train_data == 'imagenet':
                acc_sum = 0
                num_eval_batch = Y_data_eval.shape[0] // 50
                for n in range(num_eval_batch):
                    batch_offset = n * train_batchsize
                    batch_end = (n + 1) * train_batchsize
                    batch_eval_list = image_list_eval[batch_offset:batch_end]
                    feature_eval_batch = load_imagenet_raw(imagenet_test_img_path, batch_eval_list, img_height, img_width)
                    label_eval_batch = Y_data_eval[batch_offset:batch_end]
                    acc_batch = sess.run(evalOp, feed_dict={features: feature_eval_batch, labels: label_eval_batch})
                    acc_sum += acc_batch

                acc_arg = acc_sum / num_eval_batch
            else:
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

    parser.add_argument('-m', '--model', required=True, action='store',
                        choices=['resnet', 'mobilenet', 'densenet', 'mlp', 'scn'],
                        help='model type [resnet, mobilenet, mlp, densenet, scn]')

    parser.add_argument('-l', '--layer', required=True, action='store', type=int,
                        help='the number of layer decides a model, for example, the layer is 50 for resnet-50')

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

    parser.add_argument('-d', '--device', action='store', type=str, default='gpu:0', choices=['gpu:0', 'gpu:1'],
                        help='select a device to run device')

    args = parser.parse_args()

    if args.model in ['resnet'] and args.layer not in [18, 34, 50, 101, 152]:
        raise ValueError('number of layers are not supported in ResNet')
    elif args.model in ['densenet'] and args.layer not in [121, 169, 201, 264]:
        raise ValueError('number of layers are not supported in DenseNet')
    else:
        train_layer = args.layer

    train_data = args.train_set
    train_model = args.model
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
        imagenet_train_label_path = cfg_path_yml.imagenet_t50k_label_path
        imagenet_test_img_path = cfg_path_yml.imagenet_t1k_img_raw_path
        imagenet_test_label_path = cfg_path_yml.imagenet_t1k_label_path

        Y_data_train = load_imagenet_labels_onehot(imagenet_train_label_path, num_classes)
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
    train_op, eval_op, train_model_name = build_model()

    train_eval_model(train_op, eval_op, train_model_name)
