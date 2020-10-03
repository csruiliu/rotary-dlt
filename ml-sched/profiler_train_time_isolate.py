import tensorflow as tf
import argparse
from timeit import default_timer as timer

from model_importer import ModelImporter
import config_path as cfg_path_yml
import config_parameter as cfg_para_yml
from utils_img_func import load_imagenet_labels_onehot, load_cifar10_keras, load_mnist_label_onehot, load_mnist_image


def profile_steptime(model_info_args):
    model_info = model_info_args.replace('leaky_relu', 'leakyrelu')
    hyperparameter_list = model_info.split('_')
    print(hyperparameter_list)
    job_id = hyperparameter_list[0]
    model_type = hyperparameter_list[1]
    model_layer = int(hyperparameter_list[2])
    batch_size = int(hyperparameter_list[3])
    model_optimizer = hyperparameter_list[4]
    learning_rate = float(hyperparameter_list[5])
    if hyperparameter_list[6] == 'leakyrelu':
        model_activation = 'leaky_relu'
    else:
        model_activation = hyperparameter_list[6]

    train_dataset = hyperparameter_list[7]

    img_width = 0
    img_height = 0
    num_classes = 0
    num_channels = 0

    if train_dataset == 'imagenet':
        print('train the model on imagenet')
        img_width = cfg_para_yml.img_width_imagenet
        img_height = cfg_para_yml.img_height_imagenet
        num_channels = cfg_para_yml.num_channels_rgb
        num_classes = cfg_para_yml.num_class_imagenet

        imagenet_train_img_path = cfg_path_yml.imagenet_t50k_img_raw_path
        imagenet_train_label_path = cfg_path_yml.imagenet_t50k_label_path
        imagenet_test_img_path = cfg_path_yml.imagenet_t1k_img_raw_path
        imagenet_test_label_path = cfg_path_yml.imagenet_t1k_label_path

        train_label = load_imagenet_labels_onehot(imagenet_train_label_path, num_classes)
        eval_label = load_imagenet_labels_onehot(imagenet_test_label_path, num_classes)

    elif train_dataset == 'cifar10':
        print('train the model on cifar10')
        img_width = cfg_para_yml.img_width_cifar10
        img_height = cfg_para_yml.img_height_cifar10
        num_channels = cfg_para_yml.num_channels_rgb
        num_classes = cfg_para_yml.num_class_cifar10

        cifar10_path = cfg_path_yml.cifar_10_path
        train_data, train_label, eval_data, eval_label = load_cifar10_keras()

    elif train_dataset == 'mnist':
        print('train the model on mnist')
        img_width = cfg_para_yml.img_width_mnist
        img_height = cfg_para_yml.img_height_mnist
        num_channels = cfg_para_yml.num_channels_bw
        num_classes = cfg_para_yml.num_class_mnist

        mnist_train_img_path = cfg_path_yml.mnist_train_img_path
        mnist_train_label_path = cfg_path_yml.mnist_train_label_path
        mnist_test_img_path = cfg_path_yml.mnist_test_10k_img_path
        mnist_test_label_path = cfg_path_yml.mnist_test_10k_label_path

        train_data = load_mnist_image(mnist_train_img_path)
        train_label = load_mnist_label_onehot(mnist_train_label_path)
        eval_data = load_mnist_image(mnist_test_img_path)
        eval_label = load_mnist_label_onehot(mnist_test_label_path)

    features = tf.placeholder(tf.float32, [None, img_width, img_height, num_channels])
    labels = tf.placeholder(tf.int64, [None, num_classes])

    dm = ModelImporter(model_type, job_id, model_layer, img_height, img_width, num_channels,
                       num_classes, batch_size, model_optimizer, learning_rate, model_activation, False)
    model_entity = dm.get_model_entity()
    model_logit = model_entity.build(input_features=features, is_training=True)

    train_step = model_entity.train(model_logit, labels)

    #########################################################################
    # Traing the model
    #########################################################################

    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True

    step_time = 0
    step_count = 0
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        num_batch = train_label.shape[0] // batch_size
        for i in range(num_batch):
            start_time = timer()
            #print('step {} / {}'.format(i + 1, num_batch))
            batch_offset = i * batch_size
            batch_end = (i + 1) * batch_size

            X_mini_batch_feed = train_data[batch_offset:batch_end]
            Y_mini_batch_feed = train_label[batch_offset:batch_end]

            sess.run(train_step, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})

            end_time = timer()
            dur_time = end_time - start_time
            #print("step time:", dur_time)
            step_time += dur_time
            step_count += 1

    avg_step_time = step_time / step_count * 1000

    return avg_step_time


if __name__ == '__main__':

    #########################################################################
    # Constant parameters
    #########################################################################

    rand_seed = 10000

    #########################################################################
    # Parameters read from command
    #########################################################################

    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--result_path', required=True, action='store', help='the path of a result file')

    args = parser.parse_args()
    result_path = args.result_path

    train_time_list = list()
    step_num_list = list()

    with open(result_path) as fp:
        line = fp.readline()
        while line:
            if line.startswith('**Job Result**: '):
                line_trim = line.replace('**Job Result**: ', '')
                model_info, result_info = line_trim.split(',')
                step = result_info.split('_')[1]
                avg_time = profile_steptime(model_info)
                train_time_list.append(avg_time)
                step_num_list.append(step)
                #total_time += avg_time * int(step)
            line = fp.readline()
    #print('total time: {}'.format(total_time))
    print(train_time_list)
    print('#########################')
    print(step_num_list)
