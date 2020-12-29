import tensorflow as tf
from enum import Enum
import numpy as np
import time
import os

from relish.common.model_importer import ModelImporter
from relish.common.dataset_loader import load_dataset_para, load_train_dataset, load_eval_dataset
from relish.tools.img_tool import load_imagenet_raw


class State(Enum):
    RUN = 1
    PAUSE = 2
    STOP = 3


class Trail:
    def __init__(self, job_conf):
        self.trail_id = job_conf['job_id']
        self.model_type = job_conf['model_type']
        self.layer_number = job_conf['model_layer_num']
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

        self.img_w, self.img_h, self.num_chn, self.num_class = load_dataset_para(self.train_dataset)
        self.train_feature, self.train_label = load_train_dataset(self.train_dataset)
        self.eval_feature, self.eval_label = load_eval_dataset(self.train_dataset)

    def train_simulate(self):
        time.sleep(np.random.randint(1, 4))
        self.train_progress += 1
        print('[process-{},trail-{}] training progress: {} epochs'.format(os.getpid(), self.trail_id,
                                                                          self.train_progress))

    def evaluate_simulate(self):
        self.cur_accuracy += np.random.uniform(0, 0.2)
        print('[process-{},trail-{}] after {} epochs, accuracy: {}'.format(os.getpid(), self.trail_id,
                                                                           self.train_progress, self.cur_accuracy))

    def train(self, assign_device):
        with tf.device(assign_device):
            feature_ph = tf.placeholder(tf.float32, [None, self.img_w, self.img_h, self.num_chn])
            label_ph = tf.placeholder(tf.int64, [None, self.num_class])

            train_model = ModelImporter(self.model_type, str(self.trail_id),
                                        self.layer_number, self.img_h,
                                        self.img_w, self.num_chn,
                                        self.num_class, self.batch_size,
                                        self.optimizer, self.learning_rate,
                                        activation='relu', batch_padding=False)

            model_entity = train_model.get_model_entity()
            model_logit = model_entity.build(feature_ph, is_training=True)
            model_train_op = model_entity.train(model_logit, label_ph)

            config = tf.ConfigProto()
            config.allow_soft_placement = True
            config.gpu_options.allow_growth = True

            if self.train_dataset == 'imagenet':
                train_data_list = sorted(os.listdir(self.train_feature))

            with tf.Session(config=config) as sess:
                num_batch = self.train_label.shape[0] // self.batch_size

                for i in range(num_batch):
                    # print('step {} / {}'.format(i + 1, num_batch))
                    batch_offset = i * self.batch_size
                    batch_end = (i + 1) * self.batch_size

                    if self.train_dataset == 'imagenet':
                        batch_list = train_data_list[batch_offset:batch_end]
                        train_data_batch = load_imagenet_raw(self.train_feature, batch_list, self.img_h, self.img_w)
                    else:
                        train_data_batch = self.train_feature[batch_offset:batch_end]

                    train_label_batch = self.train_label[batch_offset:batch_end]
                    sess.run(model_train_op, feed_dict={feature_ph: train_data_batch, label_ph: train_label_batch})

                self.train_progress += 1

    def evaluate(self, assign_device):
        with tf.device(assign_device):
            graph = tf.Graph()
            with graph.as_default():
                feature_ph = tf.placeholder(tf.float32, [None, self.img_w, self.img_h, self.num_chn])
                label_ph = tf.placeholder(tf.int64, [None, self.num_class])
                train_model = ModelImporter(self.model_type, str(self.trail_id),
                                            self.layer_number, self.img_h,
                                            self.img_w, self.num_chn,
                                            self.num_class, self.batch_size,
                                            self.optimizer, self.learning_rate,
                                            activation='relu', batch_padding=False)
                model_entity = train_model.get_model_entity()
                model_logit = model_entity.build(feature_ph, is_training=True)
                model_eval_op = model_entity.evaluate(model_logit, label_ph)

            config = tf.ConfigProto()
            config.allow_soft_placement = True
            config.gpu_options.allow_growth = True

            with tf.Session(graph=graph, config=config) as sess:
                if self.train_dataset == 'imagenet':
                    acc_sum = 0
                    num_eval_batch = self.train_label.shape[0] // 50
                    eval_data_list = sorted(os.listdir(self.eval_feature))
                    for n in range(num_eval_batch):
                        batch_offset = n * self.batch_size
                        batch_end = (n + 1) * self.batch_size
                        batch_eval_list = eval_data_list[batch_offset:batch_end]
                        feature_eval_batch = load_imagenet_raw(self.eval_feature,
                                                               batch_eval_list,
                                                               self.img_h,
                                                               self.img_w)
                        label_eval_batch = self.eval_label[batch_offset:batch_end]
                        acc_batch = sess.run(model_eval_op, feed_dict={feature_ph: feature_eval_batch,
                                                                       label_ph: label_eval_batch})
                        acc_sum += acc_batch
                else:
                    self.cur_accuracy = sess.run(model_eval_op, feed_dict={feature_ph: self.eval_feature,
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
