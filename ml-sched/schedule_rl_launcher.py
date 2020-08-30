import tensorflow as tf
from multiprocessing import Pool
from timeit import default_timer as timer
import os

import config_path as cfg_path_yml
import config_parameter as cfg_para_yml
from model_importer import ModelImporter
from utils_img_func import *


class MLSchLauncher:
    def __init__(self, sched_job_list, sched_workload, gpu_num, cpu_num, slot_time_period):
        self.schedule_job_list = sched_job_list
        self.schedule_workload = sched_workload
        self.sch_gpu_num = gpu_num
        self.sch_cpu_num = cpu_num
        self.sch_proc_num = gpu_num + cpu_num
        self.proc_pool = None
        self.sch_slot_time_period = slot_time_period

    def launch_schedule(self):
        for sch in self.schedule_job_list:
            self.proc_pool = Pool(self.sch_proc_num)
            for didx, jidx in enumerate(sch):
                if didx < self.sch_gpu_num:
                    sch_device = '/device:GPU:'+str(didx)
                else:
                    sch_device = '/device:CPU:0'

                self.proc_pool.apply_async(self.run_single_job, args=(self.sch_slot_time_period,
                                                                      self.schedule_workload[jidx],
                                                                      sch_device))
            self.proc_pool.close()
            self.proc_pool.join()
            self.proc_pool.terminate()

    @staticmethod
    def run_single_job(time_limit, sch_job, sch_device):
        proc_start_time = timer()

        with tf.device(sch_device):
            sch_job_id = sch_job[0]
            sch_model_type = sch_job[1]
            sch_batch_size = sch_job[2]
            sch_model_optimizer = sch_job[3]
            sch_model_learning_rate = sch_job[4]
            sch_model_activation = sch_job[5]
            train_dataset = sch_job[6]

            _ckpt_path = cfg_path_yml.ckpt_save_path

            if train_dataset == 'imagenet':
                _image_path_raw = cfg_path_yml.imagenet_t10k_img_raw_path
                _image_path_bin = cfg_path_yml.imagenet_t10k_label_path
                _label_path = cfg_path_yml.imagenet_t1k_label_path
                _img_width = cfg_para_yml.img_width_imagenet
                _img_height = cfg_para_yml.img_height_imagenet
                _num_channels = cfg_para_yml.num_channels_rgb
                _num_classes = cfg_para_yml.num_class_imagenet
                train_image_list = sorted(os.listdir(_image_path_raw))
                train_label = load_imagenet_labels_onehot(_label_path, _num_channels)

            elif train_dataset == 'cifar10':
                _image_path = cfg_path_yml.cifar_10_path
                _img_width = cfg_para_yml.img_height_cifar10
                _img_height = cfg_para_yml.img_height_cifar10
                _num_channels = cfg_para_yml.num_channels_rgb
                _num_classes = cfg_para_yml.num_class_cifar10
                train_label = load_cifar10_test(_image_path)

            else:
                raise NameError('dataset cannot be found')

            model_ckpt_save_path = _ckpt_path + '/' + sch_model_type + '_' + str(sch_job_id) + '/model.ckpt'
            saver = tf.train.Saver()

            features = tf.placeholder(tf.float32, [None, _img_width, _img_height, _num_channels])
            labels = tf.placeholder(tf.int64, [None, _num_classes])

            dm = ModelImporter(sch_model_type, str(sch_job_id), 1, _img_height, _img_width, _num_channels,
                               _num_classes, sch_batch_size, sch_model_optimizer, sch_model_learning_rate,
                               sch_model_activation, False)

            model_entity = dm.get_model_entity()
            model_logit = model_entity.build(features)
            train_ops = model_entity.train(model_logit, labels)

            tf_config = tf.ConfigProto()
            tf_config.gpu_options.allow_growth = True
            tf_config.allow_soft_placement = True

            with tf.Session(config=tf_config) as sess:
                if os.path.exists(model_ckpt_save_path):
                    saver.restore(sess, model_ckpt_save_path)
                else:
                    sess.run(tf.global_variables_initializer())

                num_batch = train_label.shape[0] // sch_batch_size

                epoch_count = 0
                while True:
                    for i in range(num_batch):
                        print('*JOB at {0}*: job id {1}, model {2}-{3}, step {4}, epoch {5}'.format(sch_device, sch_job_id,
                                                                                                    sch_model_type, sch_batch_size,
                                                                                                    i, epoch_count))

                        batch_offset = i * sch_batch_size
                        batch_end = (i + 1) * sch_batch_size
                        batch_list = train_image_list[batch_offset:batch_end]
                        X_mini_batch_feed = load_imagenet_raw(_image_path_raw, batch_list, _img_height, _img_width)
                        Y_mini_batch_feed = train_label[batch_offset:batch_end, :]
                        sess.run(train_ops, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})

                        proc_end_time = timer()
                        proc_dur_time = proc_end_time - proc_start_time
                        if proc_dur_time > time_limit:
                            saver.save(sess, model_ckpt_save_path)
                            print('==============================================')
                            print('finish job {0} at device {1}'.format(sch_job, sch_device))
                            return

                    epoch_count += 1

