import numpy as np
from operator import itemgetter


class CVWorkloadGenerator:
    def __init__(self,
                 workload_size,
                 cv_light_ratio,
                 cv_med_ratio,
                 cv_heavy_ratio,
                 convergence_ratio,
                 accuracy_ratio,
                 runtime_ratio,
                 deadline_ratio,
                 half_deadline_ratio,
                 one_deadline_ratio,
                 three_deadline_ratio,
                 ten_deadline_ratio,
                 day_deadline_ratio,
                 random_seed):

        # the light, medium, heavy due to the number of parameters
        self._workload_size = workload_size

        self._cv_light_ratio = cv_light_ratio
        self._cv_med_ratio = cv_med_ratio
        self._cv_heavy_ratio = cv_heavy_ratio
        self._convergence_ratio = convergence_ratio
        self._accuracy_ratio = accuracy_ratio
        self._runtime_ratio = runtime_ratio
        self._deadline_ratio = deadline_ratio

        self._half_deadline_ratio = half_deadline_ratio
        self._one_deadline_ratio = one_deadline_ratio
        self._three_deadline_ratio = three_deadline_ratio
        self._ten_deadline_ratio = ten_deadline_ratio
        self._day_deadline_ratio = day_deadline_ratio

        # for test
        self._cv_model_light_list = ['mobilenet', 'mobilenet_v2' 'squeezenet']
        self._cv_model_med_list = ['shufflenet', 'lenet', 'alexnet']
        self._cv_model_heavy_list = ['vgg', 'densenet']

        # self._cv_model_light_list = ['inception', 'mobilenet', 'mobilenet_v2', 'squeezenet', 'xception']
        # self._cv_model_med_list = ['shufflenet', 'shufflenet_v2', 'resnet', 'resnext', 'efficientnet']
        # self._cv_model_heavy_list = ['lenet', 'vgg', 'alexnet', 'zfnet', 'densenet']

        self._cv_model_list = self._cv_model_light_list + self._cv_model_med_list + self._cv_model_heavy_list

        self._convergence_list = [('convergence', 0.05), ('convergence', 0.03), ('convergence', 0.01),
                                  ('convergence', 0.005), ('convergence', 0.003), ('convergence', 0.001),
                                  ('convergence', 0.0005), ('convergence', 0.0003), ('convergence', 0.0001)]

        self._accuracy_list = [('accuracy', 0.8), ('accuracy', 0.82), ('accuracy', 0.84), ('accuracy', 0.86),
                               ('accuracy', 0.88), ('accuracy', 0.9), ('accuracy', 0.92), ('accuracy', 0.94),
                               ('accuracy', 0.96), ('accuracy', 0.98), ('accuracy', 0.99)]

        # for test
        self._half_deadline_list = [('deadline', 75)]
        self._one_deadline_list = [('deadline', 150)]
        self._three_deadline_list = [('deadline', 450)]
        self._ten_deadline_list = [('deadline', 1500)]
        self._day_deadline_list = [('deadline', 3600)]

        # unit of deadline: second
        self._half_deadline_list = [('deadline', 1800)]
        self._one_deadline_list = [('deadline', 3600)]
        self._three_deadline_list = [('deadline', 10800)]
        self._ten_deadline_list = [('deadline', 36000)]
        self._day_deadline_list = [('deadline', 86400)]

        # unit of runtime: epoch
        self._runtime_list = [('runtime', 1), ('runtime', 5), ('runtime', 10),
                              ('runtime', 20), ('runtime', 30), ('runtime', 50)]

        self._epoch_list = [10, 20, 30, 40, 50]

        self._batch_size_list = [25, 32, 50, 64, 100, 128]

        self._opt_list = ['SGD', 'Adam', 'Adagrad', 'Momentum']

        self._learn_rate_list = [0.1, 0.01, 0.001, 0.0001, 0.00001]

        self._random_seed = random_seed

    @staticmethod
    def _random_selection(item_list, item_num):
        if item_num == 0:
            return []
        random_index = np.random.choice(np.arange(len(item_list)), size=item_num)

        res_list = list()
        for idx in random_index:
            res_list.append(item_list[idx])

        return res_list

    def generate_workload(self):
        np.random.seed(self._random_seed)

        cv_light_num = round(self._cv_light_ratio * self._workload_size)
        cv_med_num = round(self._cv_med_ratio * self._workload_size)
        cv_heavy_num = round(self._cv_heavy_ratio * self._workload_size)

        assert cv_light_num + cv_med_num + cv_heavy_num == self._workload_size

        convergence_num = round(self._convergence_ratio * self._workload_size)
        accuracy_num = round(self._accuracy_ratio * self._workload_size)
        runtime_num = round(self._runtime_ratio * self._workload_size)
        deadline_num = round(self._deadline_ratio * self._workload_size)

        half_deadline_num = round(self._half_deadline_ratio * deadline_num)
        one_deadline_num = round(self._one_deadline_ratio * deadline_num)
        three_deadline_num = round(self._three_deadline_ratio * deadline_num)
        ten_deadline_num = round(self._ten_deadline_ratio * deadline_num)
        day_deadline_num = round(self._day_deadline_ratio * deadline_num)

        deadline_num_sum = half_deadline_num + one_deadline_num + three_deadline_num + ten_deadline_num + day_deadline_num

        assert convergence_num + accuracy_num + runtime_num + deadline_num_sum == self._workload_size

        cv_light_list = self._random_selection(self._cv_model_light_list, cv_light_num)
        cv_med_list = self._random_selection(self._cv_model_med_list, cv_med_num)
        cv_heavy_list = self._random_selection(self._cv_model_heavy_list, cv_heavy_num)

        convergence_list = self._random_selection(self._convergence_list, convergence_num)
        accuracy_list = self._random_selection(self._accuracy_list, accuracy_num)
        runtime_list = self._random_selection(self._runtime_list, runtime_num)

        half_ddl_list = self._random_selection(self._half_deadline_list, half_deadline_num)
        one_ddl_list = self._random_selection(self._one_deadline_list, one_deadline_num)
        three_ddl_list = self._random_selection(self._three_deadline_list, three_deadline_num)
        ten_ddl_list = self._random_selection(self._ten_deadline_list, ten_deadline_num)
        day_ddl_list = self._random_selection(self._day_deadline_list, day_deadline_num)

        model_select_list = cv_light_list + cv_med_list + cv_heavy_list
        np.random.shuffle(model_select_list)

        deadline_list = half_ddl_list + one_ddl_list + three_ddl_list + ten_ddl_list + day_ddl_list

        objective_list = convergence_list + accuracy_list + runtime_list + deadline_list
        np.random.shuffle(objective_list)

        workload = list()
        for oidx, obj in enumerate(objective_list):
            job = dict()
            job['id'] = oidx
            job['model'] = model_select_list[oidx]
            job['opt'] = np.random.choice(self._opt_list, size=1)[0]
            job['learn_rate'] = np.random.choice(self._learn_rate_list, size=1)[0]
            job['batch_size'] = np.random.choice(self._batch_size_list, size=1)[0]
            job['training_data'] = 'cifar10'
            job['goal_type'] = obj[0]
            job['goal_value'] = obj[1]

            if job['goal_type'] == 'convergence' or job['goal_type'] == 'accuracy':
                job['goal_value_extra'] = np.random.choice(self._epoch_list)

            workload.append(job)
        return workload

