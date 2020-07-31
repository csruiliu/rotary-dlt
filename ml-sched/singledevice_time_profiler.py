
import tensorflow as tf
import argparse
import os
from timeit import default_timer as timer

import config_path as cfg_path_yml
from model_importer import ModelImporter
from utils_img_func import *


def run_job():
    with tf.device(train_device):
        features = tf.placeholder(tf.float32, [None, img_width, img_height, num_channels])
        labels = tf.placeholder(tf.int64, [None, num_classes])

        train_ops_pack = list()
        for i in range(model_num):
            train_model = ModelImporter(model_type, str(model_name_abbr[i]), model_layer_num, img_height, img_width,
                                        num_channels, num_classes, batch_size, optimizer, learning_rate, activation,
                                        False)
            model_entity = train_model.get_model_entity()
            model_logit = model_entity.build(features)
            train_ops = model_entity.train(model_logit, labels)
            train_ops_pack.append(train_ops)

        if use_raw_data:
            image_list = sorted(os.listdir(image_path))

        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        tf_config.allow_soft_placement = True

        print("job on {} start...".format(train_device))

        step_time = 0
        step_count = 0

        with tf.Session(config=tf_config) as sess:
            sess.run(tf.global_variables_initializer())
            num_batch = label_data.shape[0] // batch_size
            for i in range(num_batch):
                print('**JOB on {}**: {}-{}: step {} / {}'.format(train_device, model_type,
                                                                  batch_size, i + 1, num_batch))
                if (i + 1) % record_marker == 0:
                    start_time = timer()
                    batch_offset = i * batch_size
                    batch_end = (i + 1) * batch_size
                    if use_raw_data:
                        batch_list = image_list[batch_offset:batch_end]
                        X_mini_batch_feed = load_imagenet_raw(image_path, batch_list, img_height, img_width)
                    else:
                        X_mini_batch_feed = train_data[batch_offset:batch_end, :, :, :]
                    Y_mini_batch_feed = label_data[batch_offset:batch_end, :]
                    sess.run(train_ops_pack, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})
                    end_time = timer()
                    dur_time = end_time - start_time
                    print("step time:", dur_time)
                    step_time += dur_time
                    step_count += 1
                else:
                    batch_offset = i * batch_size
                    batch_end = (i + 1) * batch_size
                    if use_raw_data:
                        batch_list = image_list[batch_offset:batch_end]
                        X_mini_batch_feed = load_imagenet_raw(image_path, batch_list, img_height, img_width)
                    else:
                        X_mini_batch_feed = train_data[batch_offset:batch_end, :, :, :]
                    Y_mini_batch_feed = label_data[batch_offset:batch_end, :]
                    sess.run(train_ops_pack, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})

        print(step_time)
        print(step_count)
        print('job average step time (ms): [{}]'.format(step_time / step_count * 1000))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--train_device', required=True, action='store', choices=['cpu', 'gpu'],
                        help='train on [CPU, GPU]')
    parser.add_argument('-m', '--model_type', required=True, action='store',
                        choices=['resnet', 'mobilenet', 'densenet', 'mlp', 'scn'],
                        help='cpu model type [resnet,mobilenet,mlp,densenet]')
    parser.add_argument('-l', '--layer_num', action='store', type=int,
                        help='how many layer the model has')
    parser.add_argument('-n', '--model_num', required=True, action='store', type=int,
                        help='indicate the number of cpu model')
    parser.add_argument('-b', '--batch_size', required=True, action='store', type=int,
                        help='cpu job batch size [32,50,64,100,128]')
    parser.add_argument('-r', '--learn_rate', required=True, action='store', type=float,
                        help='learning rate of cpu model like 0.01, 0.001')
    parser.add_argument('-o', '--optimization', required=True, action='store',
                        choices=['SGD', 'Adam', 'Adagrad', 'Momentum'],
                        help='optimization of cpu model like [SGD, Adam, Adagrad, Momentum]')
    parser.add_argument('-a', '--activation', required=True, action='store',
                        choices=['relu', 'sigmoid', 'tanh', 'leaky_relu'],
                        help='activation of cpu model like [relu, sigmoid, tanh, leaky_relu]')
    parser.add_argument('-t', '--train_set', required=True, action='store', choices=['imagenet', 'cifar10'],
                        help='training set [imagenet, cifar10]')
    parser.add_argument('--bin_data', action='store_true', help='use raw data to train not binary')

    args = parser.parse_args()
    if args.train_set in ['cifar10'] and args.raw_data:
        parser.error('--bin-format can only be set when train_set=[imagenet]')
    if args.model_type not in ['mlp', 'scn'] and args.layer_num:
        parser.error('--layer_number can only be set when model_type=[mlp, scn]')

    rand_seed = 10000
    record_marker = 2

    if args.train_device == 'cpu':
        train_device = '/cpu:0'
    elif args.train_device == 'gpu':
        train_device = '/gpu:0'
    else:
        train_device = None

    if args.layer_num:
        model_layer_num = args.layer_num
    else:
        model_layer_num = 1

    model_type = args.model_type
    model_num = args.model_num
    batch_size = args.batch_size
    learning_rate = args.learn_rate
    optimizer = args.optimization
    activation = args.activation
    train_dataset = args.train_set
    use_bin_data = args.bin_data

    if train_dataset == 'imagenet':
        img_width = 224
        img_height = 224
        num_channels = 3
        num_classes = 1000

        if use_bin_data:
            image_path = cfg_path_yml.imagenet_t1k_bin_path
            train_data = load_imagenet_bin(image_path, num_channels, img_width, img_height)
        else:
            image_path = cfg_path_yml.imagenet_t1k_img_path
        label_path = cfg_path_yml.imagenet_t1k_label_path
        label_data = load_imagenet_labels_onehot(label_path, num_classes)

    elif train_dataset == 'cifar10':
        img_width = 32
        img_height = 32
        num_channels = 3
        num_classes = 10
        train_data, label_data = load_cifar_train(cfg_path_yml.cifar_10_path, rand_seed)

    model_name_abbr = np.random.choice(rand_seed, model_num, replace=False).tolist()

    run_job()
