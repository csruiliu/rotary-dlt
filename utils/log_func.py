from timeit import default_timer as timer
from datetime import datetime
import multiprocessing as mp

log_counter = mp.Value('i', 0)
log_time = mp.Value('d', 0)


def log_time_accuracy(job_instance_key,
                      current_acc,
                      shared_time_dict,
                      shared_epoch_dict,
                      shared_accuracy_dict):
    log_counter.value += 1
    if log_counter.value == 1:
        start_time = timer()
    now_time_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    sub_runtime_list = shared_time_dict[job_instance_key]
    sub_runtime_list.append(str(shared_epoch_dict[job_instance_key]) + ':' + now_time_date)
    shared_time_dict[job_instance_key] = sub_runtime_list

    sub_accuracy_list = shared_accuracy_dict[job_instance_key]
    sub_accuracy_list.append(str(current_acc) + ':' + now_time_date)
    shared_accuracy_dict[job_instance_key] = sub_accuracy_list

    if log_counter.value == 1:
        end_time = timer()
        log_time.value = end_time - start_time


def log_start_eval(job_name, pid, device):
    print('====== [EVALUATION START] job {} at process {} on device {} ======'.format(job_name, pid, device))


def log_end_eval(job_name, accuracy, device):
    print('====== [EVALUATION END] job {} accuracy:{} on device {} ======'.format(job_name, accuracy, device))


def log_start_train(job_name, pid, device):
    print('====== [TRAINING START] job {} at process {} on device {} ======'.format(job_name, pid, device))


def log_get_job(job_name, pid, device):
    print('###### [GET JOB] running job {} at process {} on device {} ######'.format(job_name, pid, device))

