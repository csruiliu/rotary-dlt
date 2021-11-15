import os
import logging
from timeit import default_timer as timer
from datetime import datetime
import multiprocessing as mp


_logger_instance = None

log_counter = mp.Value('i', 0)
log_time = mp.Value('d', 0)
rotary_time = mp.Value('d', 0)


def create_logger_singleton(name="rotary",
                            log_level="INFO",
                            log_file=None,
                            file_mode="a"):
    """
        file mode: the default is a, which means append
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)-8s -  %(filename)-30s:%(lineno)-4d - %(funcName)s() %(message)s"
    )

    # create file handler which logs even debug messages
    if log_file is not None:
        fid = os.path.realpath(log_file)
        fh = logging.FileHandler(fid, file_mode)
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    else:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def get_logger_instance():
    global _logger_instance

    if _logger_instance is None:
        _logger_instance = create_logger_singleton()

    return _logger_instance


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


def log_eval_job(job_name, pid, device):
    now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print('# EVALUATE JOB {}, process {}, device {}, {} #'.format(job_name, pid, device, now_time))


def log_train_job(job_name, pid, device):
    now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger = get_logger_instance()
    logger.info('# TRAIN JOB {}, process {}, device {}, {} #'.format(job_name, pid, device, now_time))


def log_get_job(job_name, pid, device):
    now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger = get_logger_instance()
    logger.info('# GET JOB {}, process {}, device {}, {} #'.format(job_name, pid, device, now_time))
