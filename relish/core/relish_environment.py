import numpy as np
import tensorflow as tf
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts

import relish.tools.reward_func as reward_function


class SchedEnv:
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
        reward = getattr(reward_function, self._reward_function)(self._accuracy_array,
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
