import numpy as np


def average_accuracy(job_return_array_list):
    return np.average(job_return_array_list)


def top_accuracy(job_return_array_list):
    return np.amax(job_return_array_list)
