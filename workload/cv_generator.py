import numpy as np
from math import ceil as ceiling
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

        self._cv_model_light_list = ['inception', 'mobilenet', 'mobilenet_v2', 'squeezenet', 'xception']
        self._cv_model_med_list = ['shufflenet', 'shufflenet_v2', 'resnet', 'resnext', 'efficientnet']
        self._cv_model_heavy_list = ['lenet', 'vgg', 'alexnet', 'zfnet', 'densenet']

        self._cv_model_list = self._cv_model_light_list + self._cv_model_med_list + self._cv_model_heavy_list

        self._convergence_list = [('convergence', 0.1), ('convergence', 0.05), ('convergence', 0.01),
                                  ('convergence', 0.005), ('convergence', 0.001),
                                  ('convergence', 0.0005), ('convergence', 0.0001)]

        self._accuracy_list = [('accuracy', 0.8), ('accuracy', 0.82), ('accuracy', 0.84), ('accuracy', 0.86),
                               ('accuracy', 0.88), ('accuracy', 0.9), ('accuracy', 0.92), ('accuracy', 0.94),
                               ('accuracy', 0.96), ('accuracy', 0.98), ('accuracy', 0.99)]

        # unit of deadline: mins
        self._deadline_list = [('deadline', 30), ('deadline', 60), ('deadline', 180), ('deadline', 300),
                               ('deadline', 600), ('deadline', 1440)]

        # unit of runtime: epoch
        self._runtime_list = [('runtime', 1), ('runtime', 5), ('runtime', 10),
                              ('runtime', 20), ('runtime', 50), ('runtime', 100)]

        self._epoch_list = [10, 20, 30, 40, 50]

        self._batch_size_list = [25, 32, 50, 64, 100, 128]

        self._opt_list = ['SGD', 'Adam', 'Adagrad', 'Momentum']

        self._learn_rate_list = [0.1, 0.01, 0.001, 0.0001, 0.00001]

        self._random_seed = random_seed

    @staticmethod
    def _random_selection(item_list, item_num):
        random_index = np.random.choice(np.arange(len(item_list)), size=item_num)
        res_list = list((itemgetter(*random_index)(item_list)))
        return res_list

    def generate_workload(self):
        np.random.seed(self._random_seed)

        cv_light_num = ceiling(self._cv_light_ratio * self._workload_size)
        cv_med_num = ceiling(self._cv_med_ratio * self._workload_size)
        cv_heavy_num = ceiling(self._cv_light_ratio * self._workload_size)

        assert cv_light_num + cv_med_num + cv_heavy_num == self._workload_size

        convergence_num = ceiling(self._convergence_ratio * self._workload_size)
        accuracy_num = ceiling(self._accuracy_ratio * self._workload_size)
        runtime_num = ceiling(self._runtime_ratio * self._workload_size)
        deadline_num = ceiling(self._deadline_ratio * self._workload_size)

        assert convergence_num + accuracy_num + runtime_num + deadline_num == self._workload_size

        cv_light_list = self._random_selection(self._cv_model_light_list, cv_light_num)
        cv_med_list = self._random_selection(self._cv_model_med_list, cv_med_num)
        cv_heavy_list = self._random_selection(self._cv_model_heavy_list, cv_heavy_num)

        convergence_list = self._random_selection(self._convergence_list, convergence_num)
        accuracy_list = self._random_selection(self._accuracy_list, accuracy_num)
        runtime_list = self._random_selection(self._runtime_list, runtime_num)
        deadline_list = self._random_selection(self._deadline_list, deadline_num)

        model_select_list = cv_light_list + cv_med_list + cv_heavy_list
        np.random.shuffle(model_select_list)
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
            job['dataset'] = 'cifar10'
            job['goal_type'] = obj[0]
            job['goal_value'] = obj[1]

            if job['goal_type'] == 'convergence' or job['goal_type'] == 'accuracy':
                job['goal_value_extra'] = np.random.choice(self._epoch_list)

            workload.append(job)
        return workload


if __name__ == "__main__":
    workload_size_arg = 20

    cv_light_ratio_arg = 0.4
    cv_med_ratio_arg = 0.2
    cv_heavy_ratio_arg = 0.4

    assert cv_light_ratio_arg + cv_med_ratio_arg + cv_heavy_ratio_arg == 1

    convergence_ratio_arg = 0.2
    accuracy_ratio_arg = 0.4
    runtime_ratio_arg = 0.2
    deadline_ratio_arg = 0.2

    assert convergence_ratio_arg + accuracy_ratio_arg + runtime_ratio_arg + deadline_ratio_arg == 1

    random_seed_arg = 10000

    wg = CVWorkloadGenerator(workload_size_arg,
                             cv_light_ratio_arg,
                             cv_med_ratio_arg,
                             cv_heavy_ratio_arg,
                             convergence_ratio_arg,
                             accuracy_ratio_arg,
                             runtime_ratio_arg,
                             deadline_ratio_arg,
                             random_seed_arg)

    dl_workload = wg.generate_workload()
    print(dl_workload)