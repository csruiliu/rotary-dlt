import numpy as np


def average_accuracy(accuracy_array, gpu_steptime_array, cpu_steptime_array):
    gpu_steptime = np.average(gpu_steptime_array)
    cpu_steptime = np.average(cpu_steptime_array)

    overhead = (gpu_steptime - cpu_steptime) / gpu_steptime
    progress = (cpu_steptime / gpu_steptime) * len(cpu_steptime_array)
    avg_accuracy = np.average(accuracy_array)
    reward = avg_accuracy - abs(overhead - progress)

    return reward


def average_accuracy_evaluation(job_return_list):
    return np.average(job_return_list)


def top_accuracy(accuracy_array, gpu_steptime_array, cpu_steptime_array):
    gpu_steptime = np.average(gpu_steptime_array)
    cpu_steptime = np.average(cpu_steptime_array)

    overhead = (gpu_steptime - cpu_steptime) / gpu_steptime
    progress = (cpu_steptime / gpu_steptime) * len(cpu_steptime_array)
    top_accuracy = np.amax(accuracy_array)
    reward = top_accuracy - abs(overhead - progress)

    return reward


def top_accuracy_evaluation(job_return_list):
    return np.amax(job_return_list)