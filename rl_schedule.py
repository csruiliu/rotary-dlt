import tensorflow as tf
from multiprocessing import Process, Manager
from timeit import default_timer as timer
import numpy as np
import os

from tf_agents.agents.ppo import ppo_agent
from tf_agents.networks import actor_distribution_network
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.replay_buffers import tf_uniform_replay_buffer
import tf_agents.specs.tensor_spec as tensor_spec
from tf_agents.trajectories import time_step as ts

from utils.dataset_tool import load_imagenet_raw
from utils.dataset_tool import load_dataset_para
from utils.dataset_tool import load_eval_dataset
from utils.dataset_tool import load_train_dataset
from utils.workload_func import generate_workload_slo
from estimator.accuracy_estimator import AccuracyEstimator
import config.config_parameter as cfg_para
import config.config_path as cfg_path

tf.compat.v1.enable_v2_behavior()


class RLEngine:
    def __init__(self, sch_env, reward_function):
        self._mlsch_env = sch_env
        self._rl_agent = None
        self._replay_buffer = None
        self._reward_list = list()
        self._reward_func = reward_function

    def build_sch_agent(self):

        actor_net = actor_distribution_network.ActorDistributionNetwork(self._mlsch_env.observation_spec(),
                                                                        self._mlsch_env.action_spec(),
                                                                        fc_layer_params=(100,))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_step_counter = tf.Variable(0)

        self._rl_agent = ppo_agent.PPOAgent(self._mlsch_env.time_step_spec(), self._mlsch_env.action_spec(),
                                            actor_net=actor_net, optimizer=optimizer, normalize_rewards=True,
                                            train_step_counter=train_step_counter)

        self._rl_agent.initialize()
        self._rl_agent.train = common.function(self._rl_agent.train)
        self._rl_agent.train_step_counter.assign(0)

    def train_sch_agent(self, num_train_iterations, collect_episodes_per_iteration, steps_num_per_batch,
                        log_interval, include_eval, eval_interval, num_eval_episodes_for_train):
        self._init_replay_buffer(steps_num_per_batch)

        for i in range(num_train_iterations):
            print('training iteration {0}'.format(i))
            # Collect a few step using collect_policy and save to the replay buffer.
            self._replay_collect_episode(collect_episodes_per_iteration)

            experience = self._replay_buffer.gather_all()
            train_loss = self._rl_agent.train(experience)
            self._replay_buffer.clear()

            step = self._rl_agent.train_step_counter.numpy()

            if step % log_interval == 0:
                print('step = {0}: loss = {1}'.format(step, train_loss.loss))

            if include_eval and step % eval_interval == 0:
                reward = self.evaluate_sch_agent(num_eval_episodes_for_train)
                print('evaluate the agent at step = {0}: Average Return = {1}'.format(step, reward))
                self._reward_list.append(reward)

        return self._reward_list

    def evaluate_sch_agent(self, eval_num_episodes):
        print('evaluate the agent for {} episodes.'.format(eval_num_episodes))
        episode_return_list = list()

        for _ in range(eval_num_episodes):
            time_step = self._mlsch_env.reset()

            while not time_step.is_last():
                action_step = self._rl_agent.policy.action(time_step)
                time_step = self._mlsch_env.step(action_step.action)
                episode_return_list.append(time_step.reward)

        # avg_reward = total_return / eval_num_episodes
        print('episode_return_list', episode_return_list)
        print('###############################')
        reward = getattr(self._reward_func, self._mlsch_env.evaluation_function())(episode_return_list)

        return reward

    def generate_schedule(self):
        print('generate the schedule...')
        sch_action_list = list()
        time_step = self._mlsch_env.reset()
        while not time_step.is_last():
            action_step = self._rl_agent.policy.action(time_step)
            sch_action_list.append(action_step.action.numpy())
            time_step = self._mlsch_env.step(action_step.action)
        return sch_action_list

    def benchmark_before_training(self, benchmark_num_episodes=10):
        # Evaluate the agent's policy once before training.
        avg_return = self.evaluate_sch_agent(benchmark_num_episodes)
        self._reward_list.append(avg_return)
        print('Returns before training:{}'.format(self._reward_list))

    def _init_replay_buffer(self, steps_num_per_batch):
        self._replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(data_spec=self._rl_agent.collect_data_spec,
                                                                             batch_size=self._mlsch_env.batch_size(),
                                                                             max_length=steps_num_per_batch)

    def _replay_collect_episode(self, collect_num_episodes):
        episode_counter = 0

        while episode_counter < collect_num_episodes:
            current_time_step = self._mlsch_env.current_time_step()
            action_step = self._rl_agent.collect_policy.action(current_time_step)
            next_time_step = self._mlsch_env.step(action_step.action)
            traj = trajectory.from_transition(current_time_step, action_step, next_time_step)

            # Add trajectory to the replay buffer
            if self._replay_buffer is None:
                raise TypeError('replay buffer has not been initialize')

            self._replay_buffer.add_batch(traj)

            if traj.is_boundary():
                self._mlsch_env.reset()
                episode_counter += 1

    def _replay_collect_step(self, environment, policy, num_steps):
        step_counter = 0

        while step_counter < num_steps:
            current_time_step = environment.current_time_step()
            action_step = policy.action(current_time_step)
            next_time_step = environment.simulated_step(action_step.action)
            traj = trajectory.from_transition(current_time_step, action_step, next_time_step)

            # Add trajectory to the replay buffer
            self._replay_buffer.add_batch(traj)

            step_counter += 1

    def reset(self):
        self._rl_agent = None
        self._replay_buffer = None
        self._reward_list.clear()


class RLEnv:
    def __init__(self,
                 time_slots_num,
                 gpu_device_num,
                 cpu_device_num,
                 workload,
                 reward_function,
                 is_simulation=False):

        self._time_slots_num = time_slots_num
        self._gpu_device_num = gpu_device_num
        self._cpu_device_num = cpu_device_num
        self._workload = workload
        self._workload_size = len(workload)
        self._total_device_num = self._gpu_device_num + self._cpu_device_num
        # gpu idx: from 0 to gpu_device_num-1, cpu idx: from cpu_device_num to total_device_num-1
        self._action_spec = tensor_spec.BoundedTensorSpec(self._total_device_num, dtype=tf.int32, minimum=1,
                                                          maximum=self._workload_size, name='action')
        # observation: the accuracy and overhead of each job in the workload
        self._observation_spec = tensor_spec.TensorSpec(self._workload_size, dtype=tf.float32)
        self._time_step_spec = ts.time_step_spec(self._observation_spec)
        self._reward_function = reward_function
        self._evaluation_function = reward_function + '_evaluation'
        self._is_simulation = is_simulation

        self._current_time_step = None
        # accuracy is workload-oriented so the shape of accuracy array is based on self._workload_size
        self._accuracy_array = np.zeros(self._workload_size, dtype=np.float32)
        # steptime array records the steptime of jobs on each devices, each device has only one job
        self._steptime_array = np.zeros(shape=(self._time_slots_num, self._total_device_num), dtype=np.float32)
        self._episode_ended = False
        self._assigned_time_slots_num = 0
        self._batch_size = 1

        self._steptime_estimator = None
        self._accuracy_estimator = None

    def step(self, action):
        if self._episode_ended:
            return self.reset()

        if self._is_simulation:
            for gidx in range(self._gpu_device_num):
                job_idx = action[gidx]
                self._accuracy_array[job_idx] += np.random.uniform(0, 0.1, 1)
                self._steptime_array[self._assigned_time_slots_num][gidx] += np.random.uniform(0, 0.8, 1)

            for cidx in range(self._gpu_device_num, self._total_device_num):
                job_idx = action[cidx]
                self._accuracy_array[job_idx] += np.random.uniform(0, 0.1, 1)
                self._steptime_array[self._assigned_time_slots_num][cidx] += np.random.uniform(0, 0.8, 1)

        else:
            for gidx in range(self._gpu_device_num):
                job_idx = action[gidx]
                #self._observation_array[job_idx][] += self._accuracy_estimator.predict_gpu_steptime(workload)

        self._assigned_time_slots_num += 1

        # use award function
        reward = getattr(self._reward_function)(self._accuracy_array,
                                                self._steptime_array[:self._gpu_device_num],
                                                self._steptime_array[self._gpu_device_num:self._total_device_num])

        if self._assigned_time_slots_num == self._time_slots_num:
            self._current_time_step = ts.termination(self._accuracy_array, reward)
            self._episode_ended = True
            return self._current_time_step
        else:
            self._current_time_step = ts.transition(self._accuracy_array, reward)
            return self._current_time_step

    def load_estimator(self, mte, ae):
        self._steptime_estimator = mte
        self._accuracy_estimator = ae

    def current_time_step(self):
        return self._current_time_step

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def time_step_spec(self):
        return self._time_step_spec

    def set_batch_size(self, batch_size):
        self._batch_size = batch_size

    def batch_size(self):
        return self._batch_size

    def reset(self):
        self._episode_ended = False
        self._assigned_time_slots_num = 0
        self._accuracy_array.fill(0)
        self._steptime_array.fill(0)
        self._current_time_step = ts.restart(np.zeros(self._time_step_spec.observation.shape, dtype=np.float32))
        return self._current_time_step

    def reward_function(self):
        return self._reward_function

    def evaluation_function(self):
        return self._evaluation_function


def produce_job_roundrobin(sch_workload_use):
    global is_cover_workload
    if len(sch_workload_use) != 0:
        cur_job = sch_workload_use.pop()
        if len(sch_workload_use) == 0:
            is_cover_workload = True
        return cur_job


def schedule_job_roundrobin(sch_device_num, sch_workload):
    sch_list = list()
    for i in range(sch_device_num):
        job = produce_job_roundrobin(sch_workload)
        sch_list.append(job)
    return sch_list


def build_model(job_data, ph_features, ph_labels):
    train_dataset = cfg_para.train_dataset
    img_w, img_h, num_chn, num_cls = load_dataset_para(train_dataset)
    train_model = ModelImporter(job_data['model_type'],
                                str(job_data['job_id']),
                                job_data['model_layer_num'],
                                img_h,
                                img_w,
                                num_chn,
                                num_cls,
                                job_data['batch_size'],
                                job_data['optimizer'],
                                job_data['learning_rate'],
                                job_data['activation'],
                                batch_padding=False)

    model_entity = train_model.get_model_entity()
    model_logit = model_entity.build(ph_features, is_training=True)
    model_train_op = model_entity.train(model_logit, ph_labels)
    model_eval_op = model_entity.evaluate(model_logit, ph_labels)

    model_name = '{}_{}_{}_{}_{}_{}_{}_{}'.format(job_data['job_id'], job_data['model_type'],
                                                  job_data['model_layer_num'], job_data['batch_size'],
                                                  job_data['optimizer'], job_data['learning_rate'],
                                                  job_data['activation'], job_data['train_dataset'])

    return model_train_op, model_eval_op, model_name


def run_job(job_data, job_progress_dict, assign_device):
    start_time = timer()

    time_slots_num = cfg_para.sch_time_slots_num
    job_num = cfg_para.slo_job_num
    slot_time_period = cfg_para.sch_slot_time_period
    ckpt_save_path = cfg_path.ckpt_save_path + '/workload_' + str(job_num) + '_timeslot_' + str(time_slots_num)

    train_dataset = cfg_para.train_dataset
    img_w, img_h, num_chn, num_cls = load_dataset_para(train_dataset)
    train_feature_input, train_label_input = load_train_dataset(train_dataset)

    with tf.device(assign_device):
        features = tf.placeholder(tf.float32, [None, img_w, img_h, num_chn])
        labels = tf.placeholder(tf.int64, [None, num_cls])
        train_ops, _, model_name = build_model(job_data, features, labels)
        saver = tf.train.Saver()

        model_ckpt_save_path = ckpt_save_path + '/' + model_name
        if not os.path.exists(model_ckpt_save_path):
            os.makedirs(model_ckpt_save_path)

        checkpoint_file = model_ckpt_save_path + '/' + 'model_ckpt'
        train_batchsize = job_data['batch_size']

        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.gpu_options.allow_growth = True

        if train_dataset == 'imagenet':
            train_data_list = sorted(os.listdir(train_feature_input))

        with tf.Session(config=config) as sess:
            if os.path.isfile(checkpoint_file + '.meta'):
                saver.restore(sess, checkpoint_file)
            else:
                sess.run(tf.global_variables_initializer())

            num_batch = train_label_input.shape[0] // train_batchsize
            total_step = 0
            while True:
                for i in range(num_batch):
                    #print('step {} / {}'.format(i + 1, num_batch))
                    batch_offset = i * train_batchsize
                    batch_end = (i + 1) * train_batchsize

                    if train_dataset == 'imagenet':
                        batch_list = train_data_list[batch_offset:batch_end]
                        train_data_batch = load_imagenet_raw(train_feature_input, batch_list, img_h, img_w)
                    else:
                        train_data_batch = train_feature_input[batch_offset:batch_end]

                    train_label_batch = train_label_input[batch_offset:batch_end]

                    sess.run(train_ops, feed_dict={features: train_data_batch, labels: train_label_batch})
                    total_step += 1
                    end_time = timer()
                    dur_time = end_time - start_time
                    if dur_time > slot_time_period:
                        job_progress_dict[model_name] += total_step
                        saver.save(sess, checkpoint_file)
                        return


def schedule_job_rlsched(sch_workload, sch_reward_func):
    time_slots_num = cfg_para.sch_time_slots_num
    gpu_device_num = cfg_para.sch_gpu_num
    cpu_device_num = cfg_para.sch_cpu_num

    # init schedule environment
    mlsch_env = RLEnv(time_slots_num,
                      gpu_device_num,
                      cpu_device_num,
                      sch_workload,
                      sch_reward_func,
                      is_simulation=False)

    # Get path parameters from config
    steptime_dataset_path = cfg_path.multidevices_time_dataset_path
    accuracy_dataset_path = cfg_path.accuracy_dataset_path

    # inti schedule multi-device time estimator
    mlsch_mte = MultiDeviceTimeEstimator(top_k=3)
    mlsch_mte.import_steptime_dataset(steptime_dataset_path)

    # inti schedule multi-device accuracy estimator
    mlsch_ae = AccuracyEstimator(top_k=3)
    mlsch_ae.import_accuracy_dataset(accuracy_dataset_path)

    mlsch_env.load_estimator(mlsch_mte, mlsch_ae)

    mlsch_engine = RLEngine(mlsch_env)

    mlsch_engine.build_sch_agent()
    mlsch_engine.benchmark_before_training(benchmark_num_episodes=20)

    mlsch_engine.train_sch_agent(num_train_iterations=50,
                                 collect_episodes_per_iteration=5,
                                 steps_num_per_batch=100,
                                 log_interval=25,
                                 include_eval=False,
                                 eval_interval=50,
                                 num_eval_episodes_for_train=15)

    final_reward = mlsch_engine.evaluate_sch_agent(eval_num_episodes=20)
    print('final reward: {}'.format(final_reward))
    sch_list = mlsch_engine.generate_schedule()
    return sch_list


def evaluate_job(job_info, job_accuracy_increment_dict, job_current_accuracy_dict):
    time_slots_num = cfg_para.sch_time_slots_num
    job_num = cfg_para.slo_job_num
    ckpt_save_path = cfg_path.ckpt_save_path + '/workload_' + str(job_num) + '_timeslot_' + str(time_slots_num)

    train_dataset = cfg_para.train_dataset
    img_w, img_h, num_chn, num_cls = load_dataset_para(train_dataset)
    eval_feature_input, eval_label_input = load_eval_dataset(train_dataset)

    graph = tf.Graph()
    with graph.as_default():
        features = tf.placeholder(tf.float32, [None, img_w, img_h, num_chn])
        labels = tf.placeholder(tf.int64, [None, num_cls])
        _, eval_ops, model_name = build_model(job_info, features, labels)
        saver = tf.train.Saver()

    model_ckpt_save_path = ckpt_save_path + '/' + model_name
    checkpoint_file = os.path.join(model_ckpt_save_path, 'model_ckpt')
    train_batchsize = job_info['batch_size']

    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True

    with tf.Session(graph=graph, config=config) as sess:
        if os.path.isfile(checkpoint_file + '.meta'):
            saver.restore(sess, checkpoint_file)
        else:
            sess.run(tf.global_variables_initializer())

        if train_dataset == 'imagenet':
            acc_sum = 0
            num_eval_batch = eval_label_input.shape[0] // 50
            eval_data_list = sorted(os.listdir(eval_feature_input))
            for n in range(num_eval_batch):
                batch_offset = n * train_batchsize
                batch_end = (n + 1) * train_batchsize
                batch_eval_list = eval_data_list[batch_offset:batch_end]
                feature_eval_batch = load_imagenet_raw(eval_feature_input, batch_eval_list, img_h, img_w)
                label_eval_batch = eval_label_input[batch_offset:batch_end]
                acc_batch = sess.run(eval_ops, feed_dict={features: feature_eval_batch, labels: label_eval_batch})
                acc_sum += acc_batch
            model_acc_avg = acc_sum / num_eval_batch
        else:
            model_acc_avg = sess.run(eval_ops, feed_dict={features: eval_feature_input,
                                                          labels: eval_label_input})

    job_accuracy_increment_dict[model_name] = model_acc_avg - job_current_accuracy_dict[model_name]
    job_current_accuracy_dict[model_name] = model_acc_avg

    return model_acc_avg, model_name


def relish_run():
    ##################################################
    # Key parameters
    ##################################################

    time_slots_num = cfg_para.sch_time_slots_num
    job_num = cfg_para.slo_job_num

    ##################################################
    # Generate Workload
    ##################################################

    sched_workload = generate_workload_slo(job_num, use_seed=True)
    sched_workload_use = sched_workload.copy()

    ##################################################
    # Prepare the shared dict
    ##################################################

    # record the progress of each job during a specific schedule
    job_progress_dict = Manager().dict()

    # record the progress of each job during a specific schedule
    job_accuracy_increment_dict = Manager().dict()
    job_current_accuracy_dict = Manager().dict()

    for job in sched_workload:
        model_name = '{}_{}_{}_{}_{}_{}_{}_{}'.format(job['job_id'], job['model_type'],
                                                      job['model_layer_num'], job['batch_size'],
                                                      job['optimizer'], job['learning_rate'],
                                                      job['activation'], job['train_dataset'])
        job_progress_dict[model_name] = 0
        job_accuracy_increment_dict[model_name] = 0
        job_current_accuracy_dict[model_name] = 0


    ##################################################
    # Schedule Parameter
    ##################################################

    gpu_device_num = cfg_para.sch_gpu_num
    cpu_device_num = cfg_para.sch_cpu_num
    sched_device_num = gpu_device_num + cpu_device_num

    # reward function
    reward_function = cfg_para.slo_reward_function
    print("Reward Function: {}".format(reward_function))

    ##################################################
    # Reinforcement Learning Schedule
    ##################################################
    is_cover_workload = False
    time_slot_count = 0

    while time_slot_count < time_slots_num:
        if is_cover_workload:
            print('starting the rl-based scheduling')
            job_list = schedule_job_rlsched(sched_workload, reward_function)
        else:
            job_list = schedule_job_roundrobin(sched_device_num, sched_workload_use)

        proc_gpu_list = list()

        for gn in range(gpu_device_num):
            assign_gpu = '/gpu:' + str(gn)
            proc_gpu = Process(target=run_job, args=(job_list[gn], job_progress_dict, assign_gpu))
            proc_gpu_list.append(proc_gpu)
        proc_cpu = Process(target=run_job, args=(job_list[sched_device_num - 1], job_progress_dict, '/cpu:0'))

        for proc_gpu in proc_gpu_list:
            proc_gpu.start()
        proc_cpu.start()

        time_slot_count += 1

    sch_job_attainment_list = list()
    sch_job_name_list = list()
    sch_job_progress_list = list()

    for jidx in sched_workload:
        job_accuracy, job_name = evaluate_job(jidx)
        sch_job_attainment_list.append(job_accuracy)
        sch_job_name_list.append(job_name)
        sch_job_progress_list.append(job_progress_dict[job_name])

    workload_acc_avg = sum(sch_job_attainment_list) / job_num

    print('#########################################################')
    print('jobs attainment in the workload:')
    for job_idx, _ in enumerate(sched_workload):
        print('**Job Result**: {}_{}_{}'.format(sch_job_name_list[job_idx], sch_job_attainment_list[job_idx],
                                                sch_job_progress_list[job_idx]))
    print('**Workload Result**: {}'.format(workload_acc_avg))
