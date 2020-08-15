#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from tf_agents.trajectories import trajectory
from tf_agents.replay_buffers import tf_uniform_replay_buffer

import config_parameter as cfg_para_yml
import config_path as cfg_path_yml

from schedule_ml_environment import MLSchEnv
from schedule_ml_engine import MLSchEngine
from utils_workload_func import generate_workload

tf.compat.v1.enable_v2_behavior()


if __name__ == "__main__":

    ######################################################
    # Get general parameters from config
    ######################################################

    _rand_seed = cfg_para_yml.rand_seed
    _record_marker = cfg_para_yml.record_marker
    _use_raw_image = cfg_para_yml.use_raw_image
    _use_measure_step = cfg_para_yml.measure_step

    ########################################
    # Get workload parameters from config
    ########################################

    _sch_gpu_device_num = cfg_para_yml.sch_gpu_num
    _sch_cpu_device_num = cfg_para_yml.sch_cpu_num
    _sch_job_num = cfg_para_yml.sch_job_num
    _sch_time_slots_num = cfg_para_yml.sch_time_slots_num
    _sch_model_type_set = cfg_para_yml.sch_model_type_set
    _sch_batch_size_set = cfg_para_yml.sch_batch_size_set
    _sch_optimizer_set = cfg_para_yml.sch_optimizer_set
    _sch_learning_rate_set = cfg_para_yml.sch_learning_rate_set
    _sch_activation_set = cfg_para_yml.sch_activation_set
    _sch_proportion_rate = cfg_para_yml.placement_proportion_rate

    ########################################
    # Get path parameters from config
    ########################################

    _image_path_raw = cfg_path_yml.imagenet_t1k_img_path
    _image_path_bin = cfg_path_yml.imagenet_t1k_bin_path
    _label_path = cfg_path_yml.imagenet_t1k_label_path

    ########################################
    # Generate workload
    ########################################

    np.random.seed(_rand_seed)

    _sch_reward_function = cfg_para_yml.sch_reward_function
    _sch_time_limit = cfg_para_yml.sch_time_limit
    _sch_wl = generate_workload(_sch_job_num, _sch_model_type_set, _sch_batch_size_set,
                                _sch_optimizer_set, _sch_learning_rate_set, _sch_activation_set)

    print("Reward Function: {}".format(_sch_reward_function))
    print("Time Limit: {}".format(_sch_time_limit))

    mlsch_env = MLSchEnv(time_slots_num=_sch_time_slots_num, gpu_device_num=_sch_gpu_device_num,
                         cpu_device_num=_sch_cpu_device_num, workload=_sch_wl, is_simulation=True)

    mlsch_engine = MLSchEngine(mlsch_env)

    mlsch_engine.build_sch_agent()
    mlsch_engine.benchmark_before_training(benchmark_num_episodes=20)
    mlsch_engine.train_sch_agent(num_train_iterations=100, collect_episodes_per_iteration=2, steps_num_per_batch=100,
                                 log_interval=25, eval_interval=50, num_eval_episodes_for_train=15)
    final_reward = mlsch_engine.evaluate_sch_agent(eval_num_episodes=20)
    print('final reward: {0}'.format(final_reward))
