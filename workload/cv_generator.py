import numpy as np


class CVWorkloadGenerator:
    def __init__(self,
                 workload_size,
                 cv_light_ratio,
                 cv_med_ratio,
                 cv_heavy_ratio,
                 convergence_ratio,
                 accuracy_ratio,
                 runtime_ratio,
                 random_seed):

        # the light, medium, heavy due to the number of parameters
        self._workload_size = workload_size

        self._cv_light_ratio = cv_light_ratio
        self._cv_med_ratio = cv_med_ratio
        self._cv_heavy_ratio = cv_heavy_ratio
        self._convergence_ratio = convergence_ratio
        self._accuracy_ratio = accuracy_ratio
        self._runtime_ratio = runtime_ratio

        # for test
        # self._cv_model_light_list = ['mobilenet', 'mobilenet_v2', 'squeezenet']
        # self._cv_model_med_list = ['shufflenet', 'shufflenet_v2', 'lenet']
        # self._cv_model_heavy_list = ['vgg', 'alexnet', 'densenet']

        self._cv_model_light_list = ['inception', 'mobilenet', 'mobilenet_v2', 'squeezenet', 'xception']
        self._cv_model_med_list = ['shufflenet', 'shufflenet_v2', 'resnet', 'resnext', 'efficientnet']
        self._cv_model_heavy_list = ['lenet', 'vgg', 'alexnet', 'zfnet', 'densenet']

        self._cv_model_list = self._cv_model_light_list + self._cv_model_med_list + self._cv_model_heavy_list

        self._convergence_list = [('convergence', 0.05), ('convergence', 0.01),
                                  ('convergence', 0.005), ('convergence', 0.001),
                                  ('convergence', 0.0005), ('convergence', 0.0001)]

        self._accuracy_list = [('accuracy', 0.8), ('accuracy', 0.82), ('accuracy', 0.84), ('accuracy', 0.86),
                               ('accuracy', 0.88), ('accuracy', 0.9), ('accuracy', 0.92), ('accuracy', 0.94),
                               ('accuracy', 0.96), ('accuracy', 0.98), ('accuracy', 0.99)]

        # unit of runtime: epoch
        # self._runtime_list = [('runtime', 1), ('runtime', 5), ('runtime', 10),
        #                       ('runtime', 20), ('runtime', 50), ('runtime', 100),
        #                       ('runtime', 200), ('runtime', 300), ('runtime', 400)]

        self._runtime_list = [('runtime', 5),
                              ('runtime', 10),
                              ('runtime', 40),
                              ('runtime', 100),
                              ('runtime', 200)]

        self._epoch_list = [10, 20, 30, 40, 50, 100]

        # select small mini-batch due to the paper
        # "Revisiting Small Batch Training for Deep Neural Networks"
        self._batch_size_list = [2, 4, 8, 16, 32]

        self._opt_list = ['SGD', 'Adam', 'Adagrad', 'Momentum']

        self._learn_rate_list = [0.1, 0.01, 0.001, 0.0001, 0.00001]

        self._random_seed = random_seed

    @staticmethod
    def _random_selection(item_list, item_num, prob=None):
        if item_num == 0:
            return []

        if prob is None:
            random_index = np.random.choice(np.arange(len(item_list)), size=item_num)
        else:
            random_index = np.random.choice(np.arange(len(item_list)), size=item_num, p=prob)

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

        assert convergence_num + accuracy_num + runtime_num == self._workload_size

        cv_light_list = self._random_selection(self._cv_model_light_list, cv_light_num)
        cv_med_list = self._random_selection(self._cv_model_med_list, cv_med_num)
        cv_heavy_list = self._random_selection(self._cv_model_heavy_list, cv_heavy_num)

        convergence_list = self._random_selection(self._convergence_list, convergence_num)
        accuracy_list = self._random_selection(self._accuracy_list, accuracy_num)
        # runtime_list = self._random_selection(self._runtime_list, runtime_num, prob=[0.1, 0.2, 0.3, 0.3, 0.1])
        runtime_list_idx = [0, 1, 1, 2, 2, 3, 3, 4]
        np.random.shuffle(runtime_list_idx)
        runtime_list = list()
        for idx in runtime_list_idx:
            runtime_list.append(self._runtime_list[idx])

        model_select_list = cv_light_list + cv_med_list + cv_heavy_list
        np.random.shuffle(model_select_list)

        objective_list = convergence_list + accuracy_list + runtime_list
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

