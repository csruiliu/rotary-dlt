import numpy as np


class WorkloadGenerator:
    def __init__(self,
                 workload_size,
                 cv_light_ratio,
                 cv_med_ratio,
                 cv_heavy_ratio,
                 nlp_light_ratio,
                 nlp_med_ratio,
                 nlp_heavy_ratio,
                 convergence_ratio,
                 accuracy_ratio,
                 runtime_ratio,
                 random_seed):

        self._workload_size = workload_size
        self._cv_light_ratio = cv_light_ratio
        self._cv_med_ratio = cv_med_ratio
        self._cv_heavy_ratio = cv_heavy_ratio
        self._nlp_light_ratio = nlp_light_ratio
        self._nlp_med_ratio = nlp_med_ratio
        self._nlp_heavy_ratio = nlp_heavy_ratio
        self._convergence_ratio = convergence_ratio
        self._accuracy_ratio = accuracy_ratio
        self._runtime_ratio = runtime_ratio
        self._random_seed = random_seed

        self._cv_model_light_list = ['inception', 'mobilenet', 'mobilenet_v2', 'squeezenet']
        self._cv_model_med_list = ['shufflenet', 'shufflenet_v2', 'resnet', 'resnext']
        self._cv_model_heavy_list = ['lenet', 'vgg', 'alexnet', 'densenet']
        self._nlp_model_light_list = ['lstm']
        self._nlp_model_med_list = ['bilstm']
        self._nlp_model_heavy_list = ['bert']

        self._cv_mdoel_list = self._cv_model_light_list + self._cv_model_med_list + self._cv_model_heavy_list
        self._lstm_model_list = self._nlp_model_light_list + self._nlp_model_med_list
        self._bert_model_list = self._nlp_model_heavy_list

        self._convergence_list = [('convergence', 0.05), ('convergence', 0.01),
                                  ('convergence', 0.005), ('convergence', 0.001),
                                  ('convergence', 0.0005), ('convergence', 0.0001)]

        self._accuracy_list = [('accuracy', 0.8), ('accuracy', 0.82), ('accuracy', 0.84), ('accuracy', 0.86),
                               ('accuracy', 0.88), ('accuracy', 0.9), ('accuracy', 0.92), ('accuracy', 0.94),
                               ('accuracy', 0.96), ('accuracy', 0.98), ('accuracy', 0.99)]

        self._runtime_list = [('runtime', 5),
                              ('runtime', 10),
                              ('runtime', 20),
                              ('runtime', 30),
                              ('runtime', 40)]

        self._pretrain_runtime_list = [1, 2, 3, 4, 5]

        self._max_epoch_list = [5, 10, 15, 20, 25, 30]

        self._cv_batch_size_list = [2, 4, 8, 16]
        self._nlp_batch_size_list = [32, 64, 128]
        self._bert_batch_size_list = [32, 64]

        self._opt_list = ['SGD', 'Adam', 'Adagrad', 'Momentum']

        self._learn_rate_list = [0.1, 0.01, 0.001, 0.0001, 0.00001]

    @staticmethod
    def _random_selection(item_list, item_num, prob=None):
        if item_num == 0:
            return []

        random_index = np.random.choice(np.arange(len(item_list)), size=item_num, p=prob)

        res_list = list()
        for idx in random_index:
            res_list.append(item_list[idx])

        return res_list

    @staticmethod
    def _runtime_selection(item_list, item_num):
        if item_num == 0:
            return []

        half_runtime = round(item_num * 0.1)
        one_runtime = round(item_num * 0.2)
        three_runtime = round(item_num * 0.3)
        ten_runtime = round(item_num * 0.3)
        day_runtime = round(item_num * 0.1)

        runtime_list = list()

        for i in range(half_runtime):
            runtime_list.append(item_list[0])

        for i in range(one_runtime):
            runtime_list.append(item_list[1])

        for i in range(three_runtime):
            runtime_list.append(item_list[2])

        for i in range(ten_runtime):
            runtime_list.append(item_list[3])

        for i in range(day_runtime):
            runtime_list.append(item_list[4])

        return runtime_list

    def generate_workload(self):
        np.random.seed(self._random_seed)

        cv_light_num = round(self._cv_light_ratio * self._workload_size)
        cv_med_num = round(self._cv_med_ratio * self._workload_size)
        cv_heavy_num = round(self._cv_heavy_ratio * self._workload_size)

        nlp_light_num = round(self._nlp_light_ratio * self._workload_size)
        nlp_med_num = round(self._nlp_med_ratio * self._workload_size)
        nlp_heavy_num = round(self._nlp_heavy_ratio * self._workload_size)

        assert (cv_light_num + cv_med_num + cv_heavy_num +
                nlp_light_num + nlp_med_num + nlp_heavy_num) == self._workload_size

        convergence_num = round(self._convergence_ratio * self._workload_size)
        accuracy_num = round(self._accuracy_ratio * self._workload_size)
        runtime_num = round(self._runtime_ratio * self._workload_size)

        assert convergence_num + accuracy_num + runtime_num == self._workload_size

        cv_light_list = self._random_selection(self._cv_model_light_list, cv_light_num)
        cv_med_list = self._random_selection(self._cv_model_med_list, cv_med_num)
        cv_heavy_list = self._random_selection(self._cv_model_heavy_list, cv_heavy_num)

        nlp_light_list = self._random_selection(self._nlp_model_light_list, nlp_light_num)
        nlp_med_list = self._random_selection(self._nlp_model_med_list, nlp_med_num)
        nlp_heavy_list = self._random_selection(self._nlp_model_heavy_list, nlp_heavy_num)

        convergence_list = self._random_selection(self._convergence_list, convergence_num)
        accuracy_list = self._random_selection(self._accuracy_list, accuracy_num)
        runtime_list = self._runtime_selection(self._runtime_list, runtime_num)

        model_select_list = cv_light_list + cv_med_list + cv_heavy_list + nlp_light_list + nlp_med_list + nlp_heavy_list
        np.random.shuffle(model_select_list)

        objective_list = convergence_list + accuracy_list + runtime_list
        np.random.shuffle(objective_list)

        workload = list()
        for oidx, obj in enumerate(objective_list):
            job = dict()
            job['id'] = oidx

            job['model'] = model_select_list[oidx]
            if job['model'] in self._cv_mdoel_list:
                job['batch_size'] = np.random.choice(self._cv_batch_size_list, size=1)[0]
                job['training_data'] = 'cifar10'
            elif job['model'] in self._lstm_model_list:
                job['batch_size'] = np.random.choice(self._nlp_batch_size_list, size=1)[0]
                job['training_data'] = 'udtreebank'
            elif job['model'] in self._bert_model_list:
                job['batch_size'] = np.random.choice(self._bert_batch_size_list, size=1)[0]
                job['training_data'] = 'stanford-lmrd'
            else:
                raise ValueError('model is not supported')

            job['opt'] = np.random.choice(self._opt_list, size=1)[0]
            job['learn_rate'] = np.random.choice(self._learn_rate_list, size=1)[0]

            job['goal_type'] = obj[0]

            if job['goal_type'] == 'runtime':
                if job['model'] in self._bert_model_list:
                    job['goal_value'] = np.random.choice(self._pretrain_runtime_list,
                                                         size=1,
                                                         p=[0.1, 0.2, 0.3, 0.3, 0.1])[0]
                else:
                    job['goal_value'] = obj[1]
            else:
                job['goal_value'] = obj[1]
                if job['model'] in self._bert_model_list:
                    job['goal_value_extra'] = 5
                else:
                    job['goal_value_extra'] = np.random.choice(self._max_epoch_list)

            workload.append(job)
        return workload

    @staticmethod
    def generate_test_workload():
        ml_workload = [{'id': 0, 'model': 'inception', 'batch_size': 2, 'training_data': 'cifar10', 'opt': 'Adagrad',
                        'learn_rate': 0.001, 'goal_type': 'convergence', 'goal_value': 0.0001, 'goal_value_extra': 3},
                       {'id': 1, 'model': 'mobilenet_v2', 'batch_size': 2, 'training_data': 'cifar10', 'opt': 'Adam',
                        'learn_rate': 0.001, 'goal_type': 'runtime', 'goal_value': 5},
                       {'id': 2, 'model': 'shufflenet_v2', 'batch_size': 16, 'training_data': 'cifar10', 'opt': 'SGD',
                        'learn_rate': 0.001, 'goal_type': 'accuracy', 'goal_value': 0.86, 'goal_value_extra': 5},
                       {'id': 3, 'model': 'bert', 'batch_size': 64, 'training_data': 'stanford-lmrd', 'opt': 'Adam',
                        'learn_rate': 0.0001, 'goal_type': 'convergence', 'goal_value': 0.05, 'goal_value_extra': 3},
                       {'id': 4, 'model': 'bilstm', 'batch_size': 128, 'training_data': 'udtreebank', 'opt': 'Adam',
                        'learn_rate': 1e-05, 'goal_type': 'runtime', 'goal_value': 10},
                       {'id': 5, 'model': 'lstm', 'batch_size': 32, 'training_data': 'udtreebank', 'opt': 'Adagrad',
                        'learn_rate': 0.001, 'goal_type': 'accuracy', 'goal_value': 0.96, 'goal_value_extra': 5},
                       {'id': 6, 'model': 'resnet', 'batch_size': 4, 'training_data': 'cifar10', 'opt': 'Adagrad',
                        'learn_rate': 0.01, 'goal_type': 'convergence', 'goal_value': 0.0001, 'goal_value_extra': 15},
                       {'id': 7, 'model': 'vgg', 'batch_size': 2, 'training_data': 'cifar10', 'opt': 'Adagrad',
                        'learn_rate': 0.01, 'goal_type': 'convergence', 'goal_value': 0.0001, 'goal_value_extra': 10}]

        return ml_workload
