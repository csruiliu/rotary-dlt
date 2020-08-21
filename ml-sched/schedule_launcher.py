import tensorflow as tf
from multiprocessing import Pool
from timeit import default_timer as timer
import time


class MLSchLauncher:
    def __init__(self, sched_job_list, sched_workload, gpu_num, cpu_num, slot_time_period):
        self.schedule_job_list = sched_job_list
        self.schedule_workload = sched_workload
        self.sch_gpu_num = gpu_num
        self.sch_cpu_num = cpu_num
        self.sch_proc_num = gpu_num + cpu_num
        self.proc_pool = None
        self.sch_slot_time_period = slot_time_period

    def launch_schedule(self):
        for sch in self.schedule_job_list:
            self.proc_pool = Pool(self.sch_proc_num)
            for didx, jidx in enumerate(sch):
                if didx < self.sch_gpu_num:
                    sch_device = '/device:GPU:'+str(didx)
                else:
                    sch_device = '/device:CPU:0'

                self.proc_pool.apply_async(self.run_single_job, args=(self.sch_slot_time_period,
                                                                      self.schedule_workload[jidx],
                                                                      sch_device))
            self.proc_pool.close()
            self.proc_pool.join()
            self.proc_pool.terminate()

    @staticmethod
    def run_single_job(time_limit, sch_job, sch_device):
        proc_start_time = timer()
        proc_dur_time = 0

        while proc_dur_time < time_limit:
            time.sleep(1)
            proc_end_time = timer()
            proc_dur_time = proc_end_time - proc_start_time

        print('finish job:{0} at device {1}'.format(sch_job, sch_device))

