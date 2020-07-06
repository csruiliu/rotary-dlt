import numpy as np
import tensorflow as tf
import argparse

import config as cfg_yml
from dnn_model import DnnModel
from img_utils import *


def build_model():
    model_name_abbr = np.random.choice(rand_seed, 1, replace=False).tolist()
    dm = DnnModel(train_model, str(model_name_abbr.pop()), train_conv_layer, img_height, img_width, num_channels,
                  num_classes, train_batchsize, train_opt, train_learn_rate, train_activation, False)
    model_entity = dm.getModelEntity()
    model_logit = model_entity.build(features)
    train_step = model_entity.train(model_logit, labels)
    eval_step = model_entity.evaluate(model_logit, labels)

    return train_step, eval_step


def train_eval_model(trainOp, evalOp):
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        num_batch = Y_data.shape[0] // train_batchsize

        for e in range(train_epoch):
            for i in range(num_batch):
                batch_offset = i * train_batchsize
                batch_end = (i+1) * train_batchsize
                X_mini_batch_feed = X_data[batch_offset:batch_end, :, :, :]
                Y_mini_batch_feed = Y_data[batch_offset:batch_end, :]
                sess.run(trainOp, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})

        acc_arg = evalOp.eval({features: X_data_eval, labels: Y_data_eval})

    print("Accuracy:", acc_arg)


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

    parser.add_argument('-cl', '--conv_layer', required=False, action='store', type=int,
                        help='number of training conv layer, for example, 1, 2, 3')

    parser.add_argument('-pl', '--pool_layer', required=False, action='store', type=int,
                        help='number of training tool layer, for example, 1, 2, 3')

    parser.add_argument('-tl', '--total_layer', required=False, action='store', type=int,
                        help='number of training total layer, for example, 1, 2, 3')

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

    if train_data == 'imagenet':
        image_path_bin = cfg_yml.imagenet_t1k_bin_path
        label_path = cfg_yml.imagenet_t1k_label_path

        img_width = cfg_yml.img_width_imagenet
        img_height = cfg_yml.img_height_imagenet
        num_channels = cfg_yml.num_channels_rgb
        num_classes = cfg_yml.num_class_imagenet

        features = tf.placeholder(tf.float32, [None, img_width, img_height, num_channels])
        labels = tf.placeholder(tf.int64, [None, num_classes])
        train_op, eval_op = build_model()

        X_data = load_imagenet_bin(image_path_bin, num_channels, img_width, img_height)
        Y_data = load_imagenet_labels_onehot(label_path, num_classes)
        X_data_eval = load_imagenet_bin_pickle(image_path_bin, num_channels, img_width, img_height)
        Y_data_eval = load_imagenet_labels_onehot(label_path, num_classes)

        train_eval_model(train_op, eval_op)

    elif train_data == 'cifar10':
        pass
    elif train_data == 'mnist':
        pass