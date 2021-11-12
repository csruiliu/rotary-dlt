import numpy as np
import json
from timeit import default_timer as timer
import os
import tensorflow as tf

from rotary.models.cv.alexnet import AlexNet
from rotary.models.cv.vgg import VGG
from rotary.models.cv.zfnet import ZFNet
from rotary.models.cv.lenet import LeNet
from rotary.models.cv.resnet import ResNet
from rotary.models.cv.xception import Xception
from rotary.models.cv.squeezenet import SqueezeNet
from rotary.models.cv.shufflenet_v2 import ShuffleNetV2
from rotary.models.cv.shufflenet import ShuffleNet
from rotary.models.cv.efficientnet import EfficientNet
from rotary.models.cv.inception import Inception
from rotary.models.cv.densenet import DenseNet
from rotary.models.cv.mobilenet import MobileNet
from rotary.models.cv.mobilenet_v2 import MobileNetV2
from rotary.models.cv.resnext import ResNeXt

import rotary.reader.cifar_reader as cifar_reader


class CVProfiler:
    def __init__(self,
                 model_name,
                 dataset,
                 batch_size,
                 optimizer,
                 learning_rate,
                 epoch,
                 profile_metric,
                 output_dir,
                 *args, **kwargs):
        self.model_name = model_name
        self.dataset = dataset
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.profile_metric = profile_metric
        self.output_dir = output_dir

        if model_name == 'resnext':
            self.model_card = kwargs['model_card']
            if self.model_card is None:
                raise ValueError('model cardinality is missing')
        elif model_name == 'shufflenetv2':
            self.model_complex = kwargs['model_complex']
            if self.model_complex is None:
                raise ValueError('model complexity is missing')
        elif model_name == 'shufflenet':
            self.model_group = kwargs['model_group']
            if self.model_group is None:
                raise ValueError('model group is missing')
        elif model_name in ['densenet', 'resnet', 'vgg']:
            self.model_layer = kwargs['model_layer']
            if self.model_layer is None:
                raise ValueError('model layer is missing')
        else:
            pass

        if self.dataset == 'cifar10':
            self.img_h = 32
            self.img_w = 32
            self.img_chn = 3
            self.num_output_classes = 10

            # load dataset
            (
                self.train_feature,
                self.train_label,
                self.eval_feature,
                self.eval_label
            ) = cifar_reader.load_cifar10_keras()

    def build_model(self):
        if self.model_name == 'alexnet':
            model = AlexNet(self.num_output_classes)
        elif self.model_name == 'densenet':
            model = DenseNet(self.model_layer, self.num_output_classes)
        elif self.model_name == 'efficientnet':
            model = EfficientNet(self.num_output_classes)
        elif self.model_name == 'inception':
            model = Inception(self.num_output_classes)
        elif self.model_name == 'lenet':
            model = LeNet(self.num_output_classes)
        elif self.model_name == 'mobilenet':
            model = MobileNet(self.num_output_classes)
        elif self.model_name == 'mobilenetv2':
            model = MobileNetV2(self.num_output_classes)
        elif self.model_name == 'resnet':
            model = ResNet(self.model_layer, self.num_output_classes)
        elif self.model_name == 'resnext':
            model = ResNeXt(self.model_card, self.num_output_classes)
        elif self.model_name == 'shufflenet':
            model = ShuffleNet(self.model_group, self.num_output_classes)
        elif self.model_name == 'shufflenetv2':
            model = ShuffleNetV2(self.model_complex, self.num_output_classes)
        elif self.model_name == 'squeezenet':
            model = SqueezeNet(self.num_output_classes)
        elif self.model_name == 'vgg':
            model = VGG(self.model_layer, self.num_output_classes)
        elif self.model_name == 'xception':
            model = Xception(self.num_output_classes)
        elif self.model_name == 'zfnet':
            model = ZFNet(self.num_output_classes)
        else:
            raise ValueError('model is unsupported')

        return model

    def train_model(self, logit):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(self.train_label, logit)
        train_loss = tf.reduce_mean(cross_entropy)
        # cross_entropy_cost = tf.reduce_mean(cross_entropy)
        # reg_loss = tf.losses.get_regularization_loss()
        # train_loss = cross_entropy_cost + reg_loss

        tf.trainable_variables()

        if self.optimizer == 'Adam':
            train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(train_loss)
        elif self.optimizer == 'SGD':
            train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(train_loss)
        elif self.optimizer == 'Adagrad':
            train_op = tf.train.AdagradOptimizer(self.learning_rate).minimize(train_loss)
        elif self.optimizer == 'Momentum':
            train_op = tf.train.MomentumOptimizer(self.learning_rate, 0.9).minimize(train_loss)
        else:
            raise ValueError('Optimizer is not recognized')

        return train_op

    def evaluate_model(self, logit):
        prediction = tf.equal(tf.argmax(logit, -1), tf.argmax(self.eval_label, -1))
        eval_op = tf.reduce_mean(tf.cast(prediction, tf.float32))

        return eval_op

    def run(self):
        model_arch = self.build_model()
        feature_ph = tf.placeholder(tf.float32, [None, self.img_h, self.img_w, self.img_chn])
        label_ph = tf.placeholder(tf.int32, [None, self.num_output_classes])
        model_logit = model_arch.build(feature_ph)

        train_op = self.train_model(model_logit)
        eval_op = self.evaluate_model(model_logit)

        ###########################################
        # count overall trainable parameters
        ###########################################

        total_parameters = 0
        for variable in tf.trainable_variables():
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters

        ###################################
        # profile the model
        ###################################

        print('Start Profiling {}'.format(self.model_name))

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            num_feature = self.train_label.shape[0]
            num_batch = num_feature // self.batch_size
            rest_feature = num_feature - self.batch_size * num_batch
            acc_record_list = list()
            time_record_list = list()

            # train the model
            for e in range(self.epoch):

                start_time = timer()

                # shuffle the training data
                shf_indices = np.arange(num_feature)
                np.random.shuffle(shf_indices)
                train_feature = train_feature[shf_indices]
                train_label = train_label[shf_indices]

                for i in range(num_batch):
                    print('epoch {} / {}, step {} / {}'.format(e + 1, self.epoch, i + 1, num_batch))

                    batch_offset = i * self.batch_size
                    batch_end = (i + 1) * self.batch_size
                    train_feature_batch = train_feature[batch_offset:batch_end]
                    train_label_batch = train_label[batch_offset:batch_end]
                    sess.run(train_op, feed_dict={feature_ph: train_feature_batch, label_ph: train_label_batch})

                if rest_feature != 0:
                    print('the rest train feature: {}, train them now'.format(rest_feature))
                    rest_feature_batch = train_feature[-rest_feature:]
                    rest_label_batch = train_label[-rest_feature:]
                    sess.run(train_op, feed_dict={feature_ph: rest_feature_batch, label_ph: rest_label_batch})
                else:
                    print('no train feature left for this epoch')

                end_time = timer()
                step_time = end_time - start_time
                time_record_list.append(step_time)

                print('start evaluation phrase')
                acc_sum = 0
                eval_batch_size = 50
                num_batch_eval = self.eval_label.shape[0] // eval_batch_size
                for i in range(num_batch_eval):
                    print('evaluation step %d / %d' % (i + 1, num_batch_eval))
                    batch_offset = i * eval_batch_size
                    batch_end = (i + 1) * eval_batch_size
                    eval_feature_batch = self.eval_feature[batch_offset:batch_end]
                    eval_label_batch = self.eval_label[batch_offset:batch_end]
                    acc_batch = sess.run(eval_op,
                                         feed_dict={feature_ph: eval_feature_batch, label_ph: eval_label_batch})
                    acc_sum += acc_batch

                acc_avg = acc_sum / num_batch_eval
                print('evaluation accuracy:{}'.format(acc_avg))
                acc_record_list.append(acc_avg)

        steptime_avg = sum(time_record_list) / len(time_record_list)

        results_path = self.output_dir + '/' + self.model_name + '_' + self.profile_metric

        if self.profile_metric == 'accuracy':
            if os.path.exists(results_path):
                with open(results_path) as f:
                    model_json_list = json.load(f)
            else:
                model_json_list = list()

            # create a dict for the conf
            model_perf_dict = dict()

            model_perf_dict['model_name'] = self.model_name
            model_perf_dict['num_parameters'] = total_parameters
            model_perf_dict['batch_size'] = self.batch_size
            model_perf_dict['opt'] = self.optimizer
            model_perf_dict['learn_rate'] = self.learning_rate
            model_perf_dict['training_data'] = self.dataset
            model_perf_dict['classes'] = self.num_output_classes
            model_perf_dict['accuracy'] = acc_record_list

            model_json_list.append(model_perf_dict)

        else:
            if os.path.exists(results_path):
                with open(results_path) as f:
                    model_json_list = json.load(f)
            else:
                model_json_list = list()

            # create a dict for the conf
            model_perf_dict = dict()

            model_perf_dict['model_name'] = self.model_name
            model_perf_dict['num_parameters'] = total_parameters
            model_perf_dict['batch_size'] = self.batch_size
            model_perf_dict['input_chn'] = train_feature.shape[-1]
            model_perf_dict['input_size'] = train_feature.shape[1] * train_feature.shape[2]
            model_perf_dict['opt'] = self.optimizer
            model_perf_dict['learn_rate'] = self.learning_rate
            model_perf_dict['training_data'] = self.dataset
            model_perf_dict['classes'] = self.num_output_classes
            model_perf_dict['steptime'] = steptime_avg

            model_json_list.append(model_perf_dict)

        with open(results_path, 'w+') as f:
            json.dump(model_json_list, f)
