import numpy as np
import tensorflow as tf
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts


class MLSchEnv:
    def __init__(self, time_slots_num, gpu_device_num, cpu_device_num, workload, is_simulation=False):
        self._time_slots_num = time_slots_num
        self._gpu_device_num = gpu_device_num
        self._cpu_device_num = cpu_device_num
        self._workload = workload
        self._workload_size = len(workload)
        self._total_device_num = self._gpu_device_num + self._cpu_device_num
        # gpu idx: from 0 to gpu_device_num-1, cpu idx: from cpu_device_num to total_device_num-1
        self._action_spec = tensor_spec.BoundedTensorSpec(shape=(self._total_device_num,), dtype=tf.int32, minimum=1,
                                                          maximum=self._workload_size, name='action')
        # observation: the accuracy of each job in the workload
        self._observation_spec = tensor_spec.TensorSpec(self._workload_size, tf.float32)
        self._time_step_spec = ts.time_step_spec(self._observation_spec)
        self.is_simulation = is_simulation

        self._current_time_step = None

        self._observation_array = np.zeros(self._workload_size, dtype=np.float32)
        self._episode_ended = False
        self._assigned_time_slots_num = 0
        self._batch_size = 1

    def step(self, action):
        if self._episode_ended:
            return self.reset()

        if self.is_simulation:
            for gidx in range(self._gpu_device_num):
                job_idx = action[gidx]
                self._observation_array[job_idx] += np.random.uniform(0, 0.1, 1)

            for cidx in range(self._gpu_device_num, self._total_device_num):
                job_idx = action[cidx]
                self._observation_array[job_idx] += np.random.uniform(0, 0.1, 1)

        self._assigned_time_slots_num += 1
        reward = np.average(self._observation_array)
        if self._assigned_time_slots_num == self._time_slots_num:
            self._current_time_step = ts.termination(self._observation_array, reward)
            return self._current_time_step
        else:
            self._current_time_step = ts.transition(self._observation_array, reward)
            return self._current_time_step

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
        self._observation_array.fill(0)
        self._current_time_step = ts.restart(np.zeros(self._time_step_spec.observation.shape, dtype=np.float32))
        return self._current_time_step

    def _reward_function(self):
        pass
