
class MLSchLauncher:
    def __init__(self, sched_job_list, sched_workload, gpu_num, cpu_num):
        print()
        self.schedule_job_list = sched_job_list
        self.schedule_workload = sched_workload
        self.sch_gpu_num = gpu_num
        self.sch_cpu_num = cpu_num
        self.sch_proc_num = gpu_num + cpu_num

    def launch_schedule(self):
        for sch in self.schedule_job_list:
            for didx, jidx in enumerate(sch):
                if didx < self.sch_gpu_num:
                    self.run_single_job(self.schedule_workload[jidx], '/device:GPU:'+str(didx))
                else:
                    self.run_single_job(self.schedule_workload[jidx], '/device:CPU:0')

                print('device id: {}'.format(didx))
                print('job id:{}'.format(jidx))

    def run_single_job(self, assign_job, assign_device_id):
        print('assign_job:{0}'.format(assign_job))
        print('assign_device:{0}'.format(assign_device_id))

