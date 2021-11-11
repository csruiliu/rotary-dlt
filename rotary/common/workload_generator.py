import numpy as np


class WorkloadGenerator:
    def __init__(self,
                 workload_size,
                 residual_ratio,
                 mobile_ratio,
                 lstm_ratio,
                 bert_ratio,
                 others_ratio,
                 convergence_ratio,
                 accuracy_ratio,
                 runtime_ratio,
                 random_seed):

        self._workload_size = workload_size
        self._residual_ratio = residual_ratio
        self._mobile_ratio = mobile_ratio
        self._lstm_ratio = lstm_ratio
        self._bert_ratio = bert_ratio
        self._others_ratio = others_ratio
        self._convergence_ratio = convergence_ratio
        self._accuracy_ratio = accuracy_ratio
        self._runtime_ratio = runtime_ratio
        self._random_seed = random_seed

        self._residual_model_list = ['resnet', 'resnext', 'densenet', 'shufflenet', 'shufflenet_v2']
        self._mobile_model_list = ['mobilenet', 'mobilenet_v2', 'efficientnet']
        self._lstm_model_list = ['lstm', 'bilstm']
        self._bert_model_list = ['bert']
        self._others_model_list = ['alexnet', 'squeezenet', 'vgg', 'zfnet', 'lenet', 'inception']

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

        self._cv_batch_size_list = [2, 4, 8, 16, 32]
        self._nlp_batch_size_list = [32, 64, 128, 256]
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

        residual_model_num = round(self._residual_ratio * self._workload_size)
        mobile_model_num = round(self._mobile_ratio * self._workload_size)
        lstm_model_num = round(self._lstm_ratio * self._workload_size)
        bert_model_num = round(self._bert_ratio * self._workload_size)
        others_model_num = round(self._others_ratio * self._workload_size)

        assert (residual_model_num + mobile_model_num + lstm_model_num +
                bert_model_num + others_model_num) == self._workload_size

        convergence_num = round(self._convergence_ratio * self._workload_size)
        accuracy_num = round(self._accuracy_ratio * self._workload_size)
        runtime_num = round(self._runtime_ratio * self._workload_size)

        assert convergence_num + accuracy_num + runtime_num == self._workload_size

        residual_model_select = self._random_selection(self._residual_model_list, residual_model_num)
        mobile_model_select = self._random_selection(self._mobile_model_list, mobile_model_num)
        lstm_model_select = self._random_selection(self._lstm_model_list, lstm_model_num)
        bert_model_select = self._random_selection(self._bert_model_list, bert_model_num)
        others_model_select = self._random_selection(self._others_model_list, others_model_num)

        convergence_list = self._random_selection(self._convergence_list, convergence_num)
        accuracy_list = self._random_selection(self._accuracy_list, accuracy_num)
        runtime_list = self._runtime_selection(self._runtime_list, runtime_num)

        model_select_list = (residual_model_select + mobile_model_select + lstm_model_select +
                             bert_model_select + others_model_select)
        np.random.shuffle(model_select_list)

        objective_list = convergence_list + accuracy_list + runtime_list
        np.random.shuffle(objective_list)

        workload = list()
        for oidx, obj in enumerate(objective_list):
            job = dict()
            job['id'] = oidx
            job['model'] = model_select_list[oidx]
            if job['model'] in self._lstm_model_list:
                job['batch_size'] = np.random.choice(self._nlp_batch_size_list, size=1)[0]
                job['training_data'] = 'udtreebank'
            elif job['model'] in self._bert_model_list:
                job['batch_size'] = np.random.choice(self._bert_batch_size_list, size=1)[0]
                job['training_data'] = 'stanford-lmrd'
            else:
                job['batch_size'] = np.random.choice(self._cv_batch_size_list, size=1)[0]
                job['training_data'] = 'cifar10'

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
                job['goal_value_extra'] = np.random.choice(self._max_epoch_list)

            workload.append(job)
        return workload
