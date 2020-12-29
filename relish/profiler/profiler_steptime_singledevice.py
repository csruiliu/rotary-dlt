import tensorflow as tf
import argparse
import numpy as np
from timeit import default_timer as timer
import os

from relish.common.model_importer import ModelImporter
from relish.common.dataset_loader import load_dataset_para, load_train_dataset, load_eval_dataset
from relish.tools.img_tool import load_imagenet_raw


def profile_model():
    with tf.device(train_device):
        feature_ph = tf.placeholder(tf.float32, [None, img_w, img_h, num_chn])
        label_ph = tf.placeholder(tf.int64, [None, num_cls])

        model_name_abbr = np.random.choice(rand_seed, train_model_num, replace=False).tolist()

        train_op_list = list()
        total_conv_layer = 0
        total_pool_layer = 0
        total_residual_layer = 0
        for i in range(train_model_num):
            dm = ModelImporter(train_model_type, str(model_name_abbr.pop()),
                               train_model_layer, img_h, img_w, num_chn,
                               num_cls, train_batchsize, train_opt,
                               train_learn_rate, train_activation, False)
            model_entity = dm.get_model_entity()
            model_logit = model_entity.build(input_features=feature_ph, is_training=True)
            train_step = model_entity.train(model_logit, label_ph)
            train_op_list.append(train_step)

            conv_layer_num, pool_layer_num, residual_layer_num = model_entity.get_layer_info()
            total_conv_layer += conv_layer_num
            total_pool_layer += pool_layer_num
            total_residual_layer += residual_layer_num

        data_cost_info = '{0}-{1}-{2}-{3}'.format(img_w, num_chn, num_cls, train_batchsize*train_model_num)
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
            image_list = sorted(os.listdir(train_feature_input))

        step_time = 0
        step_count = 0
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            num_batch = train_label_input.shape[0] // train_batchsize
            for e in range(train_epoch):
                for i in range(num_batch):

                    if i != 0:
                        start_time = timer()

                    print('epoch {} / {}, step {} / {}'.format(e+1, train_epoch, i+1, num_batch))
                    batch_offset = i * train_batchsize
                    batch_end = (i+1) * train_batchsize

                    if train_dataset == 'imagenet':
                        batch_list = image_list[batch_offset:batch_end]
                        train_feature_batch = load_imagenet_raw(train_feature_input,
                                                                batch_list,
                                                                img_h, img_w)
                    else:
                        train_feature_batch = train_feature_input[batch_offset:batch_end]

                    train_label_batch = train_label_input[batch_offset:batch_end]

                    sess.run(train_op_list, feed_dict={feature_ph: train_feature_batch,
                                                       label_ph: train_label_batch})

                    if i != 0:
                        end_time = timer()
                        dur_time = end_time - start_time
                        print("step time:", dur_time)
                        step_time += dur_time
                        step_count += 1

    avg_step_time = step_time / step_count * 1000

    print("{{\"data_cost\": \"{0}\", \"model_cost\": \"{1}\", \"tflops_cost\": \"{2}\", \"model_step_time\": {3}}}"
          .format(data_cost_info, model_layer_info, gpu_tflops_info, avg_step_time))


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

    img_w, img_h, num_chn, num_cls = load_dataset_para(train_dataset)
    train_feature_input, train_label_input = load_train_dataset(train_dataset)
    eval_feature_input, eval_label_input = load_eval_dataset(train_dataset)

    profile_model()
