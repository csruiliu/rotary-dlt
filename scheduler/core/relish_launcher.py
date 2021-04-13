import tensorflow as tf
from multiprocessing import Pool
from timeit import default_timer as timer
import os

import relish.config.config_path as cfg_path
from relish.common.model_importer import ModelImporter
from relish.common.dataset_loader import load_dataset_para, load_train_dataset
from relish.tools.img_tool import load_imagenet_raw


class SchedLauncher:
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
            job_id = sch_job['job_id']
            model_type = sch_job['model_type']
            num_layer = sch_job['model_layer_num']
            batch_size = sch_job['batch_size']
            optimizer = sch_job['optimizer']
            learning_rate = sch_job['learning_rate']
            activation = sch_job['activation']
            train_dataset = sch_job['train_dataset']

            ckpt_path = cfg_path.ckpt_save_path

            img_w, img_h, num_chn, num_cls = load_dataset_para(train_dataset)
            train_feature_input, train_label_input = load_train_dataset(train_dataset)

            model_ckpt_save_path = ckpt_path + '/' + model_type + '_' + str(job_id) + '/model.ckpt'
            saver = tf.train.Saver()

            features = tf.placeholder(tf.float32, [None, img_w, img_h, num_chn])
            labels = tf.placeholder(tf.int64, [None, num_cls])

            dm = ModelImporter(model_type, str(job_id), num_layer,
                               img_h, img_w, num_chn, num_cls,
                               batch_size, optimizer, learning_rate,
                               activation, batch_padding=False)

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

                num_batch = train_label_input.shape[0] // batch_size

                if train_dataset == 'imagenet':
                    train_data_list = sorted(os.listdir(train_feature_input))

                epoch_count = 0
                while True:
                    for i in range(num_batch):
                        print('*JOB at {0}*: job id {1}, model {2}-{3}, step {4}, epoch {5}'
                              .format(sch_device, job_id, model_type, batch_size, i, epoch_count))

                        batch_offset = i * batch_size
                        batch_end = (i + 1) * batch_size
                        batch_list = train_data_list[batch_offset:batch_end]
                        train_feature_batch = load_imagenet_raw(train_feature_input, batch_list, img_h, img_w)
                        train_label_batch = train_label_input[batch_offset:batch_end, :]
                        sess.run(train_ops, feed_dict={features: train_feature_batch,
                                                       labels: train_label_batch})

                        proc_end_time = timer()
                        proc_dur_time = proc_end_time - proc_start_time
                        if proc_dur_time > time_limit:
                            saver.save(sess, model_ckpt_save_path)
                            print('==============================================')
                            print('finish job {0} at device {1}'.format(sch_job, sch_device))
                            return

                    epoch_count += 1
