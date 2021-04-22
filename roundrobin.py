from multiprocessing import Manager, Pool
from timeit import default_timer as timer
import tensorflow as tf
import os

import config.config_path as cfg_path
import config.config_workload as cfg_workload
from workload.generator import WorkloadGenerator

from workload.tensorflow_cifar.models.resnet import ResNet
from workload.tensorflow_cifar.models.densenet import DenseNet
from workload.tensorflow_cifar.models.mobilenet_v2 import MobileNetV2
from workload.tensorflow_cifar.models.mobilenet import MobileNet
from workload.tensorflow_cifar.models.vgg import VGG
from workload.tensorflow_cifar.models.lenet import LeNet
from workload.tensorflow_cifar.models.inception import Inception
from workload.tensorflow_cifar.models.alexnet import AlexNet
from workload.tensorflow_cifar.models.resnext import ResNeXt
from workload.tensorflow_cifar.models.xception import Xception
from workload.tensorflow_cifar.models.squeezenet import SqueezeNet
from workload.tensorflow_cifar.models.zfnet import ZFNet
from workload.tensorflow_cifar.models.efficientnet import EfficientNet
from workload.tensorflow_cifar.models.shufflenet import ShuffleNet
from workload.tensorflow_cifar.models.shufflenet_v2 import ShuffleNetV2
from workload.tensorflow_cifar.tools.dataset_loader import load_cifar10_keras

import workload.tensorflow_ptb.tools.ptb_reader as ptb_reader
import workload.tensorflow_ptb.tools.model_trainer as model_trainer
from workload.tensorflow_ptb.models.nnlm import NNLM
from workload.tensorflow_ptb.models.word2vec import Word2Vec
from workload.tensorflow_ptb.models.bi_lstm import BiLSTM
from workload.tensorflow_ptb.models.textrnn import TextRNN


def build_model(job_data, n_class, feature, label):
    job_id = job_data['id']
    model_type = job['model_type']
    if model_type == 'alexnet':
        model = AlexNet(num_classes=n_class)
    elif model_type == 'densenet':
        model = DenseNet(residual_layer=121, num_classes=n_class)
    elif model_type == 'efficientnet':
        model = EfficientNet(num_classes=n_class)
    elif model_type == 'inception':
        model = Inception(num_classes=n_class)
    elif model_type == 'lenet':
        model = LeNet(num_classes=n_class)
    elif model_type == 'mobilenet':
        model = MobileNet(num_classes=n_class)
    elif model_type == 'mobilenet_v2':
        model = MobileNetV2(num_classes=n_class)
    elif model_type == 'resnet':
        model = ResNet(residual_layer=18, num_classes=n_class)
    elif model_type == 'resnext':
        model = ResNeXt(cardinality=8, num_classes=n_class)
    elif model_type == 'shufflenet':
        model = ShuffleNet(num_groups=2, num_classes=n_class)
    elif model_type == 'squeezenet':
        model = SqueezeNet(num_classes=n_class)
    elif model_type == 'vgg':
        model = VGG(conv_layer=16, num_classes=n_class)
    elif model_type == 'xception':
        model = Xception(num_classes=n_class)
    elif model_type == 'zfnet':
        model = ZFNet(num_classes=n_class)
    elif model_type == 'shufflenet_v2':
        model = ShuffleNetV2(complexity=1, num_classes=n_class)
    elif model_type == 'nnlm':
        model = NNLM(n_class=n_class, n_step=2, n_hidden=2)
    elif model_type == 'bilstm':
        model = BiLSTM(n_class=n_class, n_step=2, n_hidden=2)
    elif model_type == 'word2vec':
        model = Word2Vec(voc_size=n_class, embedding_size=2)
    elif model_type == 'textrnn':
        model = TextRNN(n_class=n_class, n_step=2, n_hidden=2)
    else:
        raise ValueError("the model type is not supported")

    logit = model.build(feature)
    train_op = model_trainer.train_model(logit, label)
    eval_op = model_trainer.eval_model(logit, label)

    job_name = str(job_id) + '-' + str(model_type)

    return train_op, eval_op, job_name


def train_job(job_data, num_gpu):
    assign_device = '/gpu:' + str(job_data['id'] % num_gpu)

    train_dataset = job_data['dataset']

    if train_dataset == 'cifar10':
        start_time = timer()
        img_w = 32
        img_h = 32
        num_chn = 3
        num_cls = 10

        train_feature, train_labels, eval_feature, eval_labels = load_cifar10_keras()
        ckpt_save_path = cfg_path.ckpt_save_path + '/job_' + str(job_data['id'])

        with tf.device(assign_device):
            feature_ph = tf.placeholder(tf.float32, [None, img_w, img_h, num_chn])
            label_ph = tf.placeholder(tf.int64, [None, num_cls])
            train_op, eval_op, model_name = build_model(job_data, feature_ph, label_ph)

            saver = tf.train.Saver()
            model_ckpt_save_path = ckpt_save_path + '/' + model_name
            if not os.path.exists(model_ckpt_save_path):
                os.makedirs(model_ckpt_save_path)
            checkpoint_file = model_ckpt_save_path + '/' + 'model_ckpt'
            train_batchsize = job_data['batch_size']

            config = tf.ConfigProto()
            config.allow_soft_placement = True
            config.gpu_options.allow_growth = True

            with tf.Session(config=config) as sess:
                if os.path.isfile(checkpoint_file + '.meta'):
                    saver.restore(sess, checkpoint_file)
                else:
                    sess.run(tf.global_variables_initializer())

                num_batch = train_labels.shape[0] // train_batchsize

                while True:
                    for i in range(num_batch):
                        print('step {} / {}'.format(i + 1, num_batch))
                        batch_offset = i * train_batchsize
                        batch_end = (i + 1) * train_batchsize

                        train_data_batch = train_feature[batch_offset:batch_end]
                        train_label_batch = train_labels[batch_offset:batch_end]

                        sess.run(train_op, feed_dict={feature_ph: train_data_batch, label_ph: train_label_batch})

                    cur_accuracy = sess.run(eval_op, feed_dict={feature_ph: eval_feature, label_ph: eval_labels})

                    end_time = timer()
                    dur_time = end_time - start_time

                    pre_accuracy = job_accuracy_dict[model_name]

                    job_time_dict[model_name] += dur_time
                    job_accuracy_dict[model_name] = cur_accuracy

                    if job_data['goal_type'] == 'accuracy':
                        if cur_accuracy >= job_data['goal_value']:
                            job_done_dict[model_name] = True
                            break
                    elif job_data['goal_type'] == 'runtime':
                        if job_time_dict[model_name] > job_data['goal_value']:
                            job_done_dict[model_name] = True
                            break
                    elif job_data['goal_type'] == 'convergence':
                        delta = cur_accuracy - pre_accuracy
                        if delta < job_data['goal_value']:
                            job_done_dict[model_name] = True
                            break
                    else:
                        raise ValueError('the job objective type is not supported')

    else:
        if job_data['model'] in ['word2vec']:
            start_time = timer()
            batch_size = job_data['batch_size']
            ptb_train_data, ptb_valid_data, _, ptb_vocab_size = ptb_reader.ptb_data_skipgram(cfg_workload.ptb_path)

            train_data = list(ptb_reader.ptb_batch_iterator_skipgrams(ptb_train_data, batch_size, ptb_vocab_size))
            eval_data = list(ptb_reader.ptb_batch_iterator_skipgrams(ptb_valid_data, batch_size, ptb_vocab_size))

            num_train_epochs = len(train_data)
            num_eval_epochs = len(eval_data)

            with tf.device(assign_device):
                feature_ph = tf.placeholder(tf.float32, shape=[None, ptb_vocab_size])
                label_ph = tf.placeholder(tf.float32, shape=[None, ptb_vocab_size])

                train_op, eval_op, model_name = build_model(job_data, feature_ph, label_ph)

                saver = tf.train.Saver()
                ckpt_save_path = cfg_path.ckpt_save_path + '/job_' + str(job_data['id'])
                model_ckpt_save_path = ckpt_save_path + '/' + model_name
                if not os.path.exists(model_ckpt_save_path):
                    os.makedirs(model_ckpt_save_path)
                checkpoint_file = model_ckpt_save_path + '/' + 'model_ckpt'

                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                config.allow_soft_placement = True

                with tf.Session(config=config) as sess:
                    if os.path.isfile(checkpoint_file + '.meta'):
                        saver.restore(sess, checkpoint_file)
                    else:
                        sess.run(tf.global_variables_initializer())

                    while True:
                        for e in range(num_train_epochs):
                            print("step {} / {}".format(e + 1, num_train_epochs))
                            train_batch = train_data[e]
                            train_input_batch = train_batch[0]
                            train_target_batch = train_batch[1]
                            sess.run(train_op, feed_dict={feature_ph: train_input_batch,
                                                          label_ph: train_target_batch})

                        sum_accuracy = 0
                        for e in range(num_eval_epochs):
                            print("evaluation eval {} / {}".format(e + 1, num_eval_epochs))
                            eval_batch = eval_data[e]
                            eval_input_batch = eval_batch[0]
                            eval_target_batch = eval_batch[1]
                            eval_batch_accuracy = sess.run(eval_op, feed_dict={feature_ph: eval_input_batch,
                                                                               label_ph: eval_target_batch})
                            sum_accuracy += eval_batch_accuracy

                        cur_accuracy = sum_accuracy / num_eval_epochs

                        end_time = timer()
                        dur_time = end_time - start_time

                        pre_accuracy = job_accuracy_dict[model_name]

                        job_time_dict[model_name] += dur_time
                        job_accuracy_dict[model_name] = cur_accuracy

                        if job_data['goal_type'] == 'accuracy':
                            if cur_accuracy >= job_data['goal_value']:
                                job_done_dict[model_name] = True
                                break
                        elif job_data['goal_type'] == 'runtime':
                            if job_time_dict[model_name] > job_data['goal_value']:
                                job_done_dict[model_name] = True
                                break
                        elif job_data['goal_type'] == 'convergence':
                            delta = cur_accuracy - pre_accuracy
                            if delta < job_data['goal_value']:
                                job_done_dict[model_name] = True
                                break
                        else:
                            raise ValueError('the job objective type is not supported')

        else:
            start_time = timer()
            num_step = 35
            batch_size = job_data['batch_size']

            ptb_train_data, ptb_valid_data, _, ptb_vocab_size = ptb_reader.ptb_data_raw(cfg_workload.ptb_path)

            train_data = list(ptb_reader.ptb_batch_iterator(ptb_train_data, batch_size, num_step, ptb_vocab_size))
            eval_data = list(ptb_reader.ptb_batch_iterator(ptb_valid_data, batch_size, num_step, ptb_vocab_size))

            num_train_epochs = len(train_data)
            num_eval_epochs = len(eval_data)

            with tf.device(assign_device):
                feature_ph = tf.placeholder(tf.float32, [None, num_step, ptb_vocab_size])
                label_ph = tf.placeholder(tf.float32, [None, ptb_vocab_size])

                train_op, eval_op, model_name = build_model(job_data, feature_ph, label_ph)

                saver = tf.train.Saver()
                ckpt_save_path = cfg_path.ckpt_save_path + '/job_' + str(job_data['id'])
                model_ckpt_save_path = ckpt_save_path + '/' + model_name
                if not os.path.exists(model_ckpt_save_path):
                    os.makedirs(model_ckpt_save_path)
                checkpoint_file = model_ckpt_save_path + '/' + 'model_ckpt'

                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                config.allow_soft_placement = True

                with tf.Session(config=config) as sess:
                    if os.path.isfile(checkpoint_file + '.meta'):
                        saver.restore(sess, checkpoint_file)
                    else:
                        sess.run(tf.global_variables_initializer())

                    while True:
                        for e in range(num_train_epochs):
                            print("step {} / {}".format(e + 1, num_train_epochs))
                            train_batch = train_data[e]
                            train_input_batch = train_batch[0]
                            train_target_batch = train_batch[1]
                            sess.run(train_op, feed_dict={feature_ph: train_input_batch,
                                                          label_ph: train_target_batch})

                        sum_accuracy = 0
                        for e in range(num_eval_epochs):
                            print("evaluation eval {} / {}".format(e + 1, num_eval_epochs))
                            eval_batch = eval_data[e]
                            eval_input_batch = eval_batch[0]
                            eval_target_batch = eval_batch[1]
                            eval_batch_accuracy = sess.run(eval_op, feed_dict={feature_ph: eval_input_batch,
                                                                               label_ph: eval_target_batch})
                            sum_accuracy += eval_batch_accuracy

                        cur_accuracy = sum_accuracy / num_eval_epochs

                        end_time = timer()
                        dur_time = end_time - start_time

                        pre_accuracy = job_accuracy_dict[model_name]

                        job_time_dict[model_name] += dur_time
                        job_accuracy_dict[model_name] = cur_accuracy

                        if job_data['goal_type'] == 'accuracy':
                            if cur_accuracy >= job_data['goal_value']:
                                job_done_dict[model_name] = True
                                break
                        elif job_data['goal_type'] == 'runtime':
                            if job_time_dict[model_name] > job_data['goal_value']:
                                job_done_dict[model_name] = True
                                break
                        elif job_data['goal_type'] == 'convergence':
                            delta = cur_accuracy - pre_accuracy
                            if delta < job_data['goal_value']:
                                job_done_dict[model_name] = True
                                break
                        else:
                            raise ValueError('the job objective type is not supported')


if __name__ == "__main__":
    n_gpu = 2

    wg = WorkloadGenerator(cfg_workload.workload_size,
                           cfg_workload.cv_light_ratio,
                           cfg_workload.cv_med_ratio,
                           cfg_workload.cv_heavy_ratio,
                           cfg_workload.nlp_light_ratio,
                           cfg_workload.nlp_med_ratio,
                           cfg_workload.nlp_heavy_ratio,
                           cfg_workload.convergence_ratio,
                           cfg_workload.accuracy_ratio,
                           cfg_workload.runtime_ratio,
                           cfg_workload.random_seed)

    ml_workload = wg.generate_workload()

    job_accuracy_dict = Manager().dict()
    job_time_dict = Manager().dict()
    job_objective_dict = Manager().dict()
    job_done_dict = Manager().dict()
    proc_pool = Pool(processes=n_gpu)

    for job in ml_workload:
        job_key = str(job['id']) + '-' + job['model']
        job_accuracy_dict[job_key] = 0
        job_time_dict[job_key] = 0
        job_objective_dict[job_key] = 0
        job_done_dict[job_key] = False

    for job in ml_workload:
        proc_pool.apply_async(func=train_job, args=(job, n_gpu,))

    proc_pool.close()
    proc_pool.join()
