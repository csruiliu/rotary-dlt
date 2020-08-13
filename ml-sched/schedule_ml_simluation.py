#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from tf_agents.networks import actor_distribution_network
from tf_agents.utils import common
from tf_agents.trajectories import trajectory
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.agents.reinforce import reinforce_agent

import config_parameter as cfg_para_yml
import config_path as cfg_path_yml

from schedule_ml_environment import MLSchEnv
from utils_workload_func import generate_workload

tf.compat.v1.enable_v2_behavior()


def evaluate_avg_reward(environment, policy, num_episodes):
    print('evaluate the algorithm for {} episodes.'.format(num_episodes))
    total_return = 0.0
    for _ in range(num_episodes):
        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.simulated_step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_reward = total_return / num_episodes
    return avg_reward


def replay_collect_episode(environment, policy, num_episodes):
    episode_counter = 0

    while episode_counter < num_episodes:
        current_time_step = environment.current_time_step()
        action_step = policy.action(current_time_step)
        next_time_step = environment.simulated_step(action_step.action)
        traj = trajectory.from_transition(current_time_step, action_step, next_time_step)

        # Add trajectory to the replay buffer
        replay_buffer.add_batch(traj)

        if traj.is_boundary():
            environment.reset()
            episode_counter += 1


def replay_collect_step(environment, policy, num_steps):
    step_counter = 0

    while step_counter < num_steps:
        current_time_step = environment.current_time_step()
        action_step = policy.action(current_time_step)
        next_time_step = environment.simulated_step(action_step.action)
        traj = trajectory.from_transition(current_time_step, action_step, next_time_step)

        # Add trajectory to the replay buffer
        replay_buffer.add_batch(traj)

        step_counter += 1


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
                         cpu_device_num=_sch_cpu_device_num, workload=_sch_wl)

    actor_net = actor_distribution_network.ActorDistributionNetwork(mlsch_env.observation_spec(),
                                                                    mlsch_env.action_spec(),
                                                                    fc_layer_params=(100,))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train_step_counter = tf.Variable(0)

    tf_agent = reinforce_agent.ReinforceAgent(mlsch_env.time_step_spec(), mlsch_env.action_spec(),
                                              actor_network=actor_net, optimizer=optimizer,
                                              normalize_returns=True, train_step_counter=train_step_counter)
    tf_agent.initialize()
    tf_agent.train = common.function(tf_agent.train)
    tf_agent.train_step_counter.assign(0)

    num_eval_episodes = 7
    # Evaluate the agent's policy once before training.
    avg_return = evaluate_avg_reward(mlsch_env, tf_agent.policy, num_eval_episodes)
    returns = [avg_return]
    print('Returns before training:{}'.format(returns))

    steps_num_per_batch = 1000

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(data_spec=tf_agent.collect_data_spec,
                                                                   batch_size=mlsch_env.batch_size(),
                                                                   max_length=steps_num_per_batch)
    iterations_num = 250
    episodes_num_per_iteration = 2

    for i in range(iterations_num):
        # Collect a few step using collect_policy and save to the replay buffer.
        replay_collect_episode(mlsch_env, tf_agent.collect_policy, episodes_num_per_iteration)

        experience = replay_buffer.gather_all()
        train_loss = tf_agent.train(experience)
        replay_buffer.clear()

        step = tf_agent.train_step_counter.numpy()

        log_interval = 25

        if step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss.loss))

        eval_interval = 50

        if step % eval_interval == 0:
            avg_return = evaluate_avg_reward(mlsch_env, tf_agent.policy, num_eval_episodes)
            print('step = {0}: Average Return = {1}'.format(step, avg_return))
            returns.append(avg_return)
