from datetime import datetime


def log_time_accuracy(job_instance_key,
                      current_acc,
                      shared_time_dict,
                      shared_epoch_dict,
                      shared_accuracy_dict):
    now_time_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    sub_runtime_list = shared_time_dict[job_instance_key]
    sub_runtime_list.append(str(shared_epoch_dict[job_instance_key]) + ':' + now_time_date)
    shared_time_dict[job_instance_key] = sub_runtime_list

    sub_accuracy_list = shared_accuracy_dict[job_instance_key]
    sub_accuracy_list.append(str(current_acc) + ':' + now_time_date)
    shared_accuracy_dict[job_instance_key] = sub_accuracy_list


def log_start_eval(job_name, pid):
    print('====== [EVALUATION START] job {} at process {} ======'.format(job_name, pid))


def log_end_eval(job_name, accuracy):
    print('====== [EVALUATION END] job {} evaluation accuracy:{} ======'.format(job_name, accuracy))


def log_get_job(job_name, pid, device):
    print('###### [GET JOB] running job {} at process {} on device {} ######'.format(job_name, pid, device))

