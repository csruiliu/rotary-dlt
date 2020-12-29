import tensorflow as tf
import argparse
import numpy as np
import os

from relish.common.model_importer import ModelImporter
from relish.common.dataset_loader import load_dataset_para, load_train_dataset, load_eval_dataset
from relish.common.dataset_loader import get_dataset_input_size
from relish.tools.img_tool import load_imagenet_raw


def profile_model():
    img_w, img_h, num_chn, num_cls = load_dataset_para(train_dataset)
    train_feature_input, train_label_input = load_train_dataset(train_dataset)
    eval_feature_input, eval_label_input = load_eval_dataset(train_dataset)

    features = tf.placeholder(tf.float32, [None, img_w, img_h, num_chn])
    labels = tf.placeholder(tf.int64, [None, num_cls])

    img_w, img_h, num_chn, num_cls = load_dataset_para(train_dataset)
    input_data_size = get_dataset_input_size(train_dataset)

    with tf.device(train_device):
        model_name_abbr = np.random.choice(rand_seed, 1, replace=False).tolist()
        dm = ModelImporter(train_model, str(model_name_abbr.pop()),
                           train_layer, img_h, img_w,
                           num_chn, num_cls, train_batchsize,
                           train_opt, train_learn_rate,
                           train_activation, False)
        model_entity = dm.get_model_entity()
        model_logit = model_entity.build(input_features=features, is_training=True)
        conv_layer_num, pool_layer_num, residual_layer_num = model_entity.get_layer_info()

        model_name_output = '{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}'.format(input_data_size, num_chn,
                                                                      num_cls, train_batchsize,
                                                                      conv_layer_num, pool_layer_num,
                                                                      residual_layer_num,
                                                                      train_learn_rate, train_opt,
                                                                      train_activation, train_epoch)
        train_op = model_entity.train(model_logit, labels)
        eval_op = model_entity.evaluate(model_logit, labels)

        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.gpu_options.allow_growth = True

        if train_dataset == 'imagenet':
            image_list_train = sorted(os.listdir(train_feature_input))
            image_list_eval = sorted(os.listdir(eval_feature_input))

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            num_batch = train_feature_input.shape[0] // train_batchsize

            for e in range(train_epoch):
                for i in range(num_batch):
                    print('epoch {} / {}, step {} / {}'.format(e+1, train_epoch, i+1, num_batch))
                    batch_offset = i * train_batchsize
                    batch_end = (i+1) * train_batchsize

                    if train_dataset == 'imagenet':
                        batch_list = image_list_train[batch_offset:batch_end]
                        train_feature_batch = load_imagenet_raw(train_feature_input,
                                                                batch_list,
                                                                img_h, img_w)
                    else:
                        train_feature_batch = train_feature_input[batch_offset:batch_end]

                    train_label_batch = train_label_input[batch_offset:batch_end]
                    sess.run(train_op, feed_dict={features: train_feature_batch, labels: train_label_batch})

            if train_dataset == 'imagenet':
                acc_sum = 0
                num_eval_batch = eval_label_input.shape[0] // 50
                for n in range(num_eval_batch):
                    batch_offset = n * train_batchsize
                    batch_end = (n + 1) * train_batchsize
                    batch_eval_list = image_list_eval[batch_offset:batch_end]
                    eval_feature_batch = load_imagenet_raw(eval_feature_input,
                                                           batch_eval_list,
                                                           img_h, img_w)
                    eval_label_batch = eval_label_input[batch_offset:batch_end]
                    acc_batch = sess.run(eval_op, feed_dict={features: eval_feature_batch,
                                                            labels: eval_label_batch})
                    acc_sum += acc_batch

                acc_arg = acc_sum / num_eval_batch
            else:
                acc_arg = sess.run(eval_op, feed_dict={features: eval_feature_input,
                                                       labels: eval_label_input})

    print("{{\"model_name\": \"{}\", \"model_accuracy\": {}}}".format(train_model, acc_arg))


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

    train_dataset = args.train_set
    train_model = args.model
    train_batchsize = args.batchsize
    train_epoch = args.epoch
    train_opt = args.opt
    train_learn_rate = args.learn_rate
    train_activation = args.activation
    train_device = args.device

    profile_model(train_model, train_dataset)
