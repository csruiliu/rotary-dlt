#!/usr/bin/env python3

import math
import numpy as np
import tensorflow as tf
import mdptoolbox

import config as cfg_yml
from schedule_ml_environment import MLSchEnv
from policy_gradient_model import PolicyGradient


def log_workload(func):
    def wrapper():
        print("log some information....")
        func_result = func()
        return func_result
    return wrapper


def generate_workload():
    sampled_model_type_list = np.random.choice(_sch_model_type_set, _sch_job_num, replace=True)
    sampled_batch_size_list = np.random.choice(_sch_batch_size_set, _sch_job_num, replace=True)
    sampled_optimizer_list = np.random.choice(_sch_optimizer_set, _sch_job_num, replace=True)
    sampled_learning_rate_list = np.random.choice(_sch_learning_rate_set, _sch_job_num, replace=True)
    sampled_activation_list = np.random.choice(_sch_activation_set, _sch_job_num, replace=True)

    sch_workload = list()

    for i in range(_sch_job_num):
        sch_model_config_dict = dict()
        sch_model_config_dict['job_id'] = i
        sch_model_config_dict['model_type'] = sampled_model_type_list[i]
        sch_model_config_dict['batch_size'] = sampled_batch_size_list[i]
        sch_model_config_dict['optimizer'] = sampled_optimizer_list[i]
        sch_model_config_dict['learning_rate'] = sampled_learning_rate_list[i]
        sch_model_config_dict['activation'] = sampled_activation_list[i]
        sch_model_config_dict['cur_accuracy'] = 0
        sch_model_config_dict['prev_accuracy'] = 0
        sch_workload.append(sch_model_config_dict)

    return sch_workload


def generate_accuracy():
    for i in range(_sch_job_num):
        _sch_wl[i]['prev_accuracy'] = _sch_wl[i]['cur_accuracy']
        acc_incremental = np.random.uniform(0, 0.1, 1)
        _sch_wl[i]['cur_accuracy'] += acc_incremental[0]


def rank_evaluate():
    if _sch_reward_function == 'highest_accuracy':
        print('using highest accuracy reward function...')
        sch_wl_sorted = sorted(_sch_wl, key=lambda x: x['cur_accuracy'], reverse=True)
    elif _sch_reward_function == 'lowest_accuracy':
        print('using lowest accuracy reward function...')
        sch_wl_sorted = sorted(_sch_wl, key=lambda x: x['cur_accuracy'], reverse=False)
    elif _sch_reward_function == 'average_accuracy':
        print('using average accuracy reward function...')
        sch_wl_sorted = sorted(_sch_wl, key=lambda x: (x['cur_accuracy']-x['prev_accuracy']), reverse=True)

    placement_slice = math.floor(_proportion_rate * _sch_job_num)
    print("The placement slice at {}".format(placement_slice))
    sch_gpu_placement_list = sch_wl_sorted[0:placement_slice]
    sch_cpu_placement_list = sch_wl_sorted[placement_slice:]

    return sch_cpu_placement_list, sch_gpu_placement_list


def schedule_policy_gradient():
    print('starting schedule based on policy gradient...')

    features = tf.placeholder(tf.float32, shape=[None, 4], name='input_x')
    obs = np.arange(12).reshape(3, 4)

    pg = PolicyGradient(4, 7)
    action = pg.build_policy_network(features)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        s1, s2, s3 = sess.run(action, feed_dict={features: obs})
        print(s1, s2, s3)


def build_sch_model(observation_size, action_space_size):
    obs_ph = tf.placeholder(tf.float32, shape=[None, observation_size, _sch_job_num], name='obs_input')
    act_ph = tf.placeholder(tf.int32, shape=[None], name='act_input')
    weights_ph = tf.placeholder(tf.float32, shape=[None], name='weights_input')



'''
def random_action_simulation():
    batch_size = 50
    epoch_num = 2
    obs_dim = 3

    for i in range(epoch_num):
        print('epoch ', i)
        obs_batch_list = list()
        act_batch_list = list()
        reward_batch_list = list()
        weights_batch_list = list()

        act_len_list = list()
        reward_eps_list = list()

        obs = _env.reset()
        obs_acc = obs['accuracy']
        obs_epochs = obs['epochs']
        obs_time = obs['time']
        obs_array = np.array([obs_acc, obs_epochs, obs_time])

        while True:
            obs_batch_list.append(obs_array)

            # generate a fake action
            action_job = np.random.choice(_sch_wl)
            action = dict()
            action['job_id'] = action_job['job_id']
            action['job_model_type'] = action_job['model_type']
            action['job_batch_size'] = action_job['batch_size']
            action['device_id'] = np.random.randint(0, 2)
            action['order_id'] = np.random.randint(0, 100)

            # store the action in to the list
            act_batch_list.append(action)

            # compute the next observation, reward, is_done according to the action
            obs, reward, done, info = _env.step(action)
            reward_eps_list.append(reward)

            # convert obs to array to fit placeholder
            obs_acc = obs['accuracy']
            obs_epochs = obs['epochs']
            obs_time = obs['time']
            obs_array = np.array([obs_acc, obs_epochs, obs_time])

            if done:
                eps_reward = sum(reward_eps_list)
                eps_reward_len = len(reward_eps_list)
                reward_batch_list.append(eps_reward)
                act_len_list.append(eps_reward_len)

                # the weight for each logprob(a|s) is R(tau)
                # so each logprob should use the total reward for the
                weights_batch_list += [eps_reward] * eps_reward_len

                # reset episode-specific variables
                obs, done, reward_eps_list = _env.reset(), False, []

                if len(obs_batch_list) > batch_size:
                    print('batch size in epoch {}: {}'.format(i, len(obs_batch_list)))
                    break
'''


if __name__ == "__main__":

    ######################################################
    # Get general parameters from config
    ######################################################

    _rand_seed = cfg_yml.rand_seed
    _record_marker = cfg_yml.record_marker
    _use_raw_image = cfg_yml.use_raw_image
    _use_measure_step = cfg_yml.measure_step

    _image_path_raw = cfg_yml.imagenet_t1k_img_path
    _image_path_bin = cfg_yml.imagenet_t1k_bin_path
    _label_path = cfg_yml.imagenet_t1k_label_path

    ########################################
    # Get workload parameters from config
    ########################################

    _sch_device_num = cfg_yml.sch_device_num
    _sch_job_num = cfg_yml.sch_job_num
    _sch_model_type_set = cfg_yml.sch_model_type_set
    _sch_batch_size_set = cfg_yml.sch_batch_size_set
    _sch_optimizer_set = cfg_yml.sch_optimizer_set
    _sch_learning_rate_set = cfg_yml.sch_learning_rate_set
    _sch_activation_set = cfg_yml.sch_activation_set

    ########################################
    # Get workload parameters from config
    ########################################

    _proportion_rate = cfg_yml.placement_proportion_rate

    ########################################
    # Generate workload
    ########################################

    np.random.seed(_rand_seed)

    _sch_wl = generate_workload()
    _sch_reward_function = cfg_yml.sch_reward_function
    _sch_time_limit = cfg_yml.sch_time_limit
    print("Reward Function: {}".format(_sch_reward_function))
    print("Time Limit: {}".format(_sch_time_limit))

    ########################################
    # Initialize Gym Custom Environment
    ########################################

    #_env = gym.make('MLSchEnv-v0')
    #_env.init_env(num_devices=_sch_device_num, num_jobs=_sch_job_num, time_limit=_sch_time_limit)

    ########################################
    # Initialize Gym-free Custom Environment
    ########################################

    _env = MLSchEnv(num_devices=_sch_device_num, num_jobs=_sch_job_num, time_limit=_sch_time_limit, is_simulate=True)

    ########################################
    # Build the scheduling model
    ########################################

    #random_action_simulation()

    #schedule_policy_gradient()


    #dur_time = 0
    #start_time = timer()

    #while dur_time < _sch_time_limit:
    #    generate_accuracy()
    #    end_time = timer()
    #    dur_time = end_time - start_time
    
    #rank_evaluate()
    #schedule_simulation()

    #pg = PolicyGradient(1, 1)
    #pg.build_policy_network()



