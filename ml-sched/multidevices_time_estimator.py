import json
import numpy as np
from sklearn.linear_model import LinearRegression


class MultiDeviceTimeEstimator:
    def __init__(self, top_k):
        self.similarity_top_k = top_k

        self.conv_model_gpu_list = list()
        self.conv_model_cpu_list = list()
        self.cpu_job_info_list = list()
        self.cpu_job_steptime_list = list()
        self.gpu_job_info_list = list()
        self.gpu_job_steptime_list = list()

        self.steptime_dataset_path = None
        self.core_estimator = None

    def import_steptime_dataset(self, dataset_path):
        self.steptime_dataset_path = dataset_path

        with open(self.steptime_dataset_path) as json_file:
            json_data = json.load(json_file)

        self.conv_model_gpu_list = json_data['steptime_model_gpu']
        self.conv_model_cpu_list = json_data['steptime_model_cpu']

        self.cpu_job_info_list, self.cpu_job_steptime_list = self._build_model_dataset(self.conv_model_cpu_list, 'CPU')
        self.gpu_job_info_list, self.gpu_job_steptime_list = self._build_model_dataset(self.conv_model_gpu_list, 'GPU')

    def predict_cpu_steptime(self, input_cpu_model_dict):
        cpu_model_similarity_list = self._compute_model_similarity(input_cpu_model_dict, self.cpu_job_info_list, 'CPU')
        similarity_cpu_model_idx_list = self._rank_model_similarity(self.cpu_job_info_list, cpu_model_similarity_list)
        selected_cpu_model_list = [self.cpu_job_info_list[i] for i in similarity_cpu_model_idx_list]
        selected_cpu_model_steptime_list = [self.cpu_job_steptime_list[i] for i in similarity_cpu_model_idx_list]

        self._train_estimator(selected_cpu_model_list, selected_cpu_model_steptime_list)

        cpu_test_list = list()
        cpu_test_list.append(list(input_cpu_model_dict.values()))
        cpu_time = self.core_estimator.predict(cpu_test_list)

        return cpu_time

    def predict_gpu_steptime(self, input_gpu_model_dict):
        gpu_model_similarity_list = self._compute_model_similarity(input_gpu_model_dict, self.gpu_job_info_list, 'GPU')
        similarity_gpu_model_idx_list = self._rank_model_similarity(self.gpu_job_info_list, gpu_model_similarity_list)
        selected_gpu_model_list = [self.gpu_job_info_list[i] for i in similarity_gpu_model_idx_list]
        selected_gpu_model_steptime_list = [self.gpu_job_steptime_list[i] for i in similarity_gpu_model_idx_list]

        self._train_estimator(selected_gpu_model_list, selected_gpu_model_steptime_list)

        gpu_test_list = list()
        gpu_test_list.append(list(input_gpu_model_dict.values()))
        gpu_time = self.core_estimator.predict(gpu_test_list)

        return gpu_time

    def _compute_model_similarity(self, center_model, candidate_models, model_type):
        assert len(center_model) == len(candidate_models[0]), 'the input model and candidate models have different format'

        curmodel_similarity_list = list()
        candidate_models_similarity_list = list()

        for model_idx in candidate_models:
            curmodel_hyperparam_similarity_list = list()
            input_binary_set = set()
            candidate_binary_set = set()

            for cfg_idx in center_model:
                if model_type == 'CPU' and cfg_idx in ('channel_num', 'data_device', 'platform'):
                    input_binary_set.add(center_model[cfg_idx])
                    candidate_binary_set.add(model_idx[cfg_idx])
                elif model_type == 'GPU' and cfg_idx in ('channel_num', 'platform'):
                    input_binary_set.add(center_model[cfg_idx])
                    candidate_binary_set.add(model_idx[cfg_idx])
                else:
                    max_x = max([center_model[cfg_idx], model_idx[cfg_idx]])
                    diff_x = np.abs(center_model[cfg_idx] - model_idx[cfg_idx])
                    input_size_similarity = 1 - diff_x / max_x
                    curmodel_hyperparam_similarity_list.append(input_size_similarity)

            curmodel_binary_similarity = self._jaccard_index(input_binary_set, candidate_binary_set)
            curmodel_similarity_list.append(curmodel_binary_similarity)
            curmodel_overall_similarity = sum(curmodel_hyperparam_similarity_list) / len(
                curmodel_hyperparam_similarity_list)
            candidate_models_similarity_list.append(curmodel_overall_similarity)

        return candidate_models_similarity_list

    def _train_estimator(self, x_train, y_train):
        assert len(x_train) == len(y_train), 'different size of feature data and label data'
        x_train_list = list()
        for xidx in x_train:
            x_train_list.append(list(xidx.values()))

        linreg = LinearRegression()
        self.core_estimator = linreg.fit(x_train_list, y_train)

    def _rank_model_similarity(self, candidate_models_info_list, candidate_models_similarity_list):
        k = self.similarity_top_k
        assert len(candidate_models_info_list) >= k, 'the number of selected models is larger than the candidate models'

        similarity_sorted_idx_list = sorted(range(len(candidate_models_similarity_list)),
                                            key=lambda m: candidate_models_similarity_list[m],
                                            reverse=True)[:k]

        return similarity_sorted_idx_list

    @staticmethod
    def _build_model_dataset(model_list, model_type):
        model_dataset = list()
        model_steptime = list()

        for model_info in model_list:
            model_info_dict = dict()

            data_cost_list = model_info['data_cost'].split('-')
            model_info_dict['input_size'] = int(data_cost_list[0])
            model_info_dict['channel_num'] = int(data_cost_list[1])
            model_info_dict['batch_size'] = int(data_cost_list[2])

            model_cost_list = model_info['model_cost'].split('-')
            model_info_dict['conv_layer_num'] = int(model_cost_list[0])
            model_info_dict['total_layer_num'] = int(model_cost_list[1])

            cpu_cost_list = model_info['cpu_cost'].split('-')
            model_info_dict['cpu_overall_input_size'] = int(cpu_cost_list[0])
            model_info_dict['cpu_overall_batch_size'] = int(cpu_cost_list[1])
            model_info_dict['cpu_overall_conv_layer'] = int(cpu_cost_list[2])
            model_info_dict['cpu_overall_total_layer'] = int(cpu_cost_list[3])

            if model_type == 'CPU':
                environment_cost_list = model_info['environment_cost'].split('-')
                model_info_dict['threads_num'] = int(environment_cost_list[0])
                model_info_dict['data_device'] = int(environment_cost_list[1])
            elif model_type == 'GPU':
                environment_cost_list = model_info['environment_cost'].split('-')
                model_info_dict['gpu_tflops'] = float(environment_cost_list[0])

            model_dataset.append(model_info_dict)

            model_step_time = model_info['step_time']
            model_steptime.append(model_step_time)

        return model_dataset, model_steptime

    @staticmethod
    def _jaccard_index(s1, s2):
        size_s1 = len(s1)
        size_s2 = len(s2)

        # Get the intersection set
        intersect = s1 & s2
        size_in = len(intersect)

        # Calculate the Jaccard index using the formula
        jaccard_idx = size_in / (size_s1 + size_s2 - size_in)

        return jaccard_idx

    @staticmethod
    def generate_test_data(test_model, test_model_type):
        input_model_split = test_model.split('-')
        input_model_dict = dict()

        input_model_dict['input_size'] = int(input_model_split[0])
        input_model_dict['channel_num'] = int(input_model_split[1])
        input_model_dict['batch_size'] = int(input_model_split[2])
        input_model_dict['conv_layer_num'] = int(input_model_split[3])
        input_model_dict['total_layer_num'] = int(input_model_split[4])
        input_model_dict['cpu_overall_input_size'] = int(input_model_split[5])
        input_model_dict['cpu_overall_batch_size'] = int(input_model_split[6])
        input_model_dict['cpu_overall_conv_layer'] = int(input_model_split[7])
        input_model_dict['cpu_overall_total_layer'] = int(input_model_split[8])

        if test_model_type == 'CPU':
            input_model_dict['threads_num'] = int(input_model_split[9])
            input_model_dict['data_device'] = int(input_model_split[10])
        elif test_model_type == 'GPU':
            input_model_dict['gpu_tflops'] = float(input_model_split[9])

        return input_model_dict

'''
if __name__ == "__main__":
    INPUT_CPU_MODEL = '32-3-32-96-161-0-0-0-0-72-1'
    INPUT_GPU_MODEL = '32-3-32-96-101-224-64-96-161-6.027'
    input_gpu_model_dict = generate_test_data(INPUT_GPU_MODEL, 'GPU')
    input_cpu_model_dict = generate_test_data(INPUT_CPU_MODEL, 'CPU')
'''