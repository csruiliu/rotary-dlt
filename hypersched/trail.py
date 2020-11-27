from models.model_importer import ModelImporter
import config.config_parameter as cfg_para
import config.config_path as cfg_path
from utils.utils_img_func import load_imagenet_raw, load_imagenet_labels_onehot, load_cifar10_keras, load_mnist_image, load_mnist_label_onehot

import tensorflow as tf
from enum import Enum
import numpy as np
import time
import os
import sys
sys.path.append(os.path.abspath(".."))


class State(Enum):
    RUN = 1
    PAUSE = 2
    STOP = 3


class Trail:
    def __init__(self, job_conf):
        self.trail_id = job_conf['job_id']
        self.model_type = job_conf['model_type']
        self.layer_number = job_conf['layer']
        self.batch_size = job_conf['batch_size']
        self.optimizer = job_conf['optimizer']
        self.learning_rate = job_conf['learning_rate']
        self.train_dataset = job_conf['train_dataset']

        # record how many training epochs has been launched
        self.train_progress = 0
        self.state = None

        # the updated accuracy
        self.cur_accuracy = 0

        #######################################
        # Prepare training dataset
        #######################################
        if self.train_dataset == 'imagenet':
            self.img_width = cfg_para.img_width_imagenet
            self.img_height = cfg_para.img_height_imagenet
            self.num_classes = cfg_para.num_class_imagenet
            self.img_channels = cfg_para.num_channels_rgb
            self.imagenet_train_data_path = cfg_path.imagenet_t10k_img_raw_path
            self.imagenet_train_label_path = cfg_path.imagenet_t10k_label_path
            self.imagenet_eval_data_path = cfg_path.imagenet_t1k_img_raw_path
            self.imagenet_eval_label_path = cfg_path.imagenet_t1k_label_path

            self.train_label = load_imagenet_labels_onehot(self.imagenet_train_label_path, self.num_classes)
            self.eval_label = load_imagenet_labels_onehot(self.imagenet_eval_label_path, self.num_classes)

        elif self.train_dataset == 'cifar10':
            self.img_width = cfg_para.img_width_cifar10
            self.img_height = cfg_para.img_height_cifar10
            self.img_num_class = cfg_para.num_class_cifar10
            self.img_channels = cfg_para.num_channels_rgb
            self.img_path = cfg_path.cifar_10_path

            self.cifar10_path = cfg_path.cifar_10_path
            self.train_data, self.train_label, self.eval_data, self.eval_label = load_cifar10_keras()

        elif self.train_dataset == 'mnist':
            self.img_width = cfg_para.img_width_mnist
            self.img_height = cfg_para.img_height_mnist
            self.img_num_class = cfg_para.num_class_imagenet
            self.img_channels = cfg_para.num_channels_bw

            self.mnist_train_img_path = cfg_path.mnist_train_img_path
            self.mnist_train_label_path = cfg_path.mnist_train_label_path
            self.mnist_test_img_path = cfg_path.mnist_test_10k_img_path
            self.mnist_test_label_path = cfg_path.mnist_test_10k_label_path

            self.train_data = load_mnist_image(self.mnist_train_img_path)
            self.train_label = load_mnist_label_onehot(self.mnist_train_label_path)
            self.eval_data = load_mnist_image(self.mnist_test_img_path)
            self.eval_label = load_mnist_label_onehot(self.mnist_test_label_path)

        else:
            raise ValueError('Only support dataset: imagenet, cifar10, mnist')

    def train_simulate(self):
        time.sleep(np.random.randint(1, 4))
        self.train_progress += 1
        print('[process-{},trail-{}] training progress: {} epochs'.format(os.getpid(), self.trail_id,
                                                                          self.train_progress))

    def evaluate_simulate(self):
        np.random.seed(time.time())
        self.cur_accuracy += np.random.uniform(0, 0.2)
        print('[process-{},trail-{}] after {} epochs, accuracy: {}'.format(os.getpid(), self.trail_id,
                                                                           self.train_progress, self.cur_accuracy))

    def train(self, assign_device):
        with tf.device(assign_device):
            feature_ph = tf.placeholder(tf.float32, [None, self.img_width, self.img_height, self.img_channels])
            label_ph = tf.placeholder(tf.int64, [None, self.img_num_class])

            train_model = ModelImporter(self.model_type, str(self.trail_id), self.layer_number, self.img_height,
                                        self.img_width, self.img_channels, self.img_num_class, self.batch_size,
                                        self.optimizer, self.learning_rate, 'relu', False)

            model_entity = train_model.get_model_entity()
            model_logit = model_entity.build(feature_ph, is_training=True)
            model_train_op = model_entity.train(model_logit, label_ph)

            config = tf.ConfigProto()
            config.allow_soft_placement = True
            config.gpu_options.allow_growth = True

            if self.train_dataset == 'imagenet':
                train_data_list = sorted(os.listdir(self.imagenet_train_data_path))

            with tf.Session(config=config) as sess:
                num_batch = self.train_label.shape[0] // self.batch_size

                for i in range(num_batch):
                    # print('step {} / {}'.format(i + 1, num_batch))
                    batch_offset = i * self.batch_size
                    batch_end = (i + 1) * self.batch_size

                    if self.train_dataset == 'imagenet':
                        batch_list = train_data_list[batch_offset:batch_end]
                        train_data_batch = load_imagenet_raw(self.imagenet_train_data_path, batch_list, self.img_height,
                                                             self.img_width)
                    else:
                        train_data_batch = self.train_data[batch_offset:batch_end]

                    train_label_batch = self.train_label[batch_offset:batch_end]
                    sess.run(model_train_op, feed_dict={feature_ph: train_data_batch, label_ph: train_label_batch})

                self.train_progress += 1

    def evaluate(self, assign_device):
        with tf.device(assign_device):
            graph = tf.Graph()
            with graph.as_default():
                feature_ph = tf.placeholder(tf.float32, [None, self.img_width, self.img_height, self.img_channels])
                label_ph = tf.placeholder(tf.int64, [None, self.img_num_class])
                train_model = ModelImporter(self.model_type, str(self.trail_id), self.layer_number, self.img_height,
                                            self.img_width, self.img_channels, self.img_num_class, self.batch_size,
                                            self.optimizer, self.learning_rate, 'relu', False)
                model_entity = train_model.get_model_entity()
                model_logit = model_entity.build(feature_ph, is_training=True)
                model_eval_op = model_entity.evaluate(model_logit, label_ph)

            config = tf.ConfigProto()
            config.allow_soft_placement = True
            config.gpu_options.allow_growth = True

            if self.train_dataset == 'imagenet':
                train_data_list = sorted(os.listdir(self.imagenet_train_data_path))

            with tf.Session(graph=graph, config=config) as sess:
                num_batch = self.train_label.shape[0] // self.batch_size

                if self.train_dataset == 'imagenet':
                    acc_sum = 0
                    num_eval_batch = self.train_label.shape[0] // 50
                    for n in range(num_eval_batch):
                        batch_offset = n * self.batch_size
                        batch_end = (n + 1) * self.batch_size
                        batch_eval_list = self.eval_data[batch_offset:batch_end]
                        feature_eval_batch = load_imagenet_raw(self.imagenet_eval_data_path, batch_eval_list,
                                                               self.img_height, self.img_width)
                        label_eval_batch = self.eval_label[batch_offset:batch_end]
                        acc_batch = sess.run(model_eval_op,
                                             feed_dict={feature_ph: feature_eval_batch, label_ph: label_eval_batch})
                        acc_sum += acc_batch
                else:
                    self.cur_accuracy = sess.run(model_eval_op, feed_dict={feature_ph: self.eval_data,
                                                                       label_ph: self.eval_label})

    def set_state(self, arg_state):
        self.state = arg_state

    def get_state(self):
        return self.state

    def get_train_progress(self):
        return self.train_progress

    def get_accuracy(self):
        return self.cur_accuracy

    def get_trail_id(self):
        return self.trail_id
