import numpy as np
from timeit import default_timer as timer
import tensorflow as tf


class MLSchEnv:
    def __init__(self, num_devices, num_jobs, time_limit, is_simulate):
        self.device_num = num_devices
        self.job_num = num_jobs
        self.time_limit = time_limit
        self.is_simulate = is_simulate

        self.accuracy_job_list = list()
        self.epoch_job_list = list()
        self.time_job_list = list()
        self.action_list = list()
        self.current_step = 0
        self.start_time = timer()
        self.end_time = timer()
        self.dur_time = 0
        self.total_time = 0

        self.output_obs = dict()
        self.output_reward = 0
        self.output_done = False
        self.output_info = list()

        if self.is_simulate:
            self.model_time_dict = {'mlp-32-imagenet': 130,
                                    'mlp-50-imagenet': 230,
                                    'mlp-64-imagenet': 280,
                                    'mlp-100-imagenet': 460,
                                    'mlp-128-imagenet': 540,
                                    'mobilenet-32-imagenet': 220,
                                    'mobilenet-50-imagenet': 390,
                                    'mobilenet-64-imagenet': 480,
                                    'mobilenet-100-imagenet': 760,
                                    'mobilenet-128-imagenet': 940,
                                    'resnet-32-imagenet': 230,
                                    'resnet-50-imagenet': 400,
                                    'resnet-64-imagenet': 490,
                                    'resnet-100-imagenet': 770,
                                    'resnet-128-imagenet': 950,
                                    'densenet-32-imagenet': 290,
                                    'densenet-50-imagenet': 460,
                                    'densenet-64-imagenet': 570,
                                    'densenet-100-imagenet': 870,
                                    'densenet-128-imagenet': 1070}

    def step(self, action):
        print('MLSch Environment Step...')
        self.current_step += 1
        if self.is_simulate:
            return self._take_action_simulate(action)

    def _take_action_simulate(self, action):
        self.action_list.append(action)
        act_job_id = action['job_id']
        act_model_type = action['job_model_type']
        act_batch_size = action['job_batch_size']
        act_device_id = action['device_id']
        act_order_id = action['order_id']

        model_time_index = act_model_type + '-' + str(act_batch_size) + '-imagenet'

        self.epoch_job_list[act_job_id] += 1
        # generate the time of the job
        job_time = np.random.uniform(self.model_time_dict[model_time_index] - 5, self.model_time_dict[model_time_index] + 5)

        self.time_job_list[act_job_id] += job_time
        self.accuracy_job_list[act_job_id] += np.random.uniform(0, (1 - self.accuracy_job_list[act_job_id]) / 20)

        self.output_obs['accuracy'] = self.accuracy_job_list
        self.output_obs['epochs'] = self.epoch_job_list
        self.output_obs['time'] = self.time_job_list

        # compute the reward according to the epochs, accuracy, time,
        self.output_reward = sum(self.accuracy_job_list) / sum(self.time_job_list)

        # simulate the running time
        self.total_time += job_time

        # if reach the
        if self.total_time > self.time_limit:
            print('total time: {}'.format(self.total_time))
            print('time limit: {}'.format(self.time_limit))
            self.output_done = True

        return self.output_obs, self.output_reward, self.output_done, self.output_info

    def _take_action(self):
        pass

    def reset(self):
        print('MLSch Environment Reset...')
        self.accuracy_job_list = [0] * self.job_num
        self.epoch_job_list = [0] * self.job_num
        self.time_job_list = [0] * self.job_num
        self.action_list.clear()
        self.current_step = 0
        self.start_time = timer()
        self.dur_time = 0
        self.total_time = 0

        self.output_obs.clear()
        self.output_reward = 0
        self.output_done = False
        self.output_info.clear()

        self.output_obs['accuracy'] = self.accuracy_job_list
        self.output_obs['epochs'] = self.epoch_job_list
        self.output_obs['time'] = self.time_job_list

        return self.output_obs

    def close(self):
        print('MLSch Environment Close...')