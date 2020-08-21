#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

import config_parameter as cfg_para_yml
import config_path as cfg_path_yml

from schedule_environment import MLSchEnv
from schedule_engine import MLSchEngine
from schedule_launcher import MLSchLauncher
from multidevices_time_estimator import MultiDeviceTimeEstimator
from accuracy_estimator import AccuracyEstimator

from utils_workload_func import generate_workload

from timeit import default_timer as timer

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
    _sch_slot_time_period = cfg_para_yml.sch_slot_time_period
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
    _steptime_dataset_path = cfg_path_yml.multidevices_time_dataset_path
    _accuracy_dataset_path = cfg_path_yml.accuracy_dataset_path

    ########################################
    # Generate workload
    ########################################

    np.random.seed(_rand_seed)

    _sch_reward_function = cfg_para_yml.sch_reward_function
    _sch_wl = generate_workload(_sch_job_num, _sch_model_type_set, _sch_batch_size_set,
                                _sch_optimizer_set, _sch_learning_rate_set, _sch_activation_set)

    print("Reward Function: {}".format(_sch_reward_function))

    # init schedule environment
    mlsch_env = MLSchEnv(time_slots_num=_sch_time_slots_num, gpu_device_num=_sch_gpu_device_num,
                         cpu_device_num=_sch_cpu_device_num, workload=_sch_wl,
                         reward_function=_sch_reward_function, is_simulation=True)

    # inti schedule multi-device time estimator
    mlsch_mte = MultiDeviceTimeEstimator(top_k=3)
    mlsch_mte.import_steptime_dataset(_steptime_dataset_path)

    # inti schedule multi-device accuracy estimator
    mlsch_ae = AccuracyEstimator(top_k=3)
    mlsch_ae.import_accuracy_dataset(_accuracy_dataset_path)

    mlsch_env.load_estimator(mlsch_mte, mlsch_ae)

    mlsch_engine = MLSchEngine(mlsch_env)

    mlsch_engine.build_sch_agent()
    mlsch_engine.benchmark_before_training(benchmark_num_episodes=20)
    start_time = timer()
    mlsch_engine.train_sch_agent(num_train_iterations=50, collect_episodes_per_iteration=5, steps_num_per_batch=100,
                                 log_interval=25, include_eval=False, eval_interval=50, num_eval_episodes_for_train=15)
    end_time = timer()
    dur_time = end_time - start_time
    print('training time: {0}'.format(dur_time))
    final_reward = mlsch_engine.evaluate_sch_agent(eval_num_episodes=20)
    print('final reward: {0}'.format(final_reward))

    sch_list = mlsch_engine.generate_schedule()

    mlsch_launcher = MLSchLauncher(sch_list, _sch_wl, _sch_gpu_device_num, _sch_cpu_device_num, _sch_slot_time_period)
    mlsch_launcher.launch_schedule()
