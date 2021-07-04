import numpy as np
import json
from scipy.optimize import curve_fit


class AccuracyEstimator:
    def __init__(self, top_k):
        self.similarity_top_k = top_k

        self.conv_model_list = None
        self.accuracy_dataset_path = None

        self.model_info_list = None
        self.model_accuracy_list = None
        self.model_epoch_list = None

    def import_accuracy_dataset(self, dataset_path):
        self.accuracy_dataset_path = dataset_path

        with open(self.accuracy_dataset_path) as json_file:
            json_data = json.load(json_file)

        self.conv_model_list = json_data['conv_model']
        self.model_info_list, self.model_accuracy_list, self.model_epoch_list = self._build_conv_model_dataset(self.conv_model_list)

    def predict_accuracy(self, input_model_dict, input_model_epoch):
        model_similarity_list = self._compute_model_similarity(input_model_dict, self.model_info_list)
        similarity_model_idx_list = self._rank_model_similarity(self.model_info_list, model_similarity_list)

        neighbor_model_accuracy_list = [self.model_accuracy_list[i] for i in similarity_model_idx_list]
        neighbor_model_epoch_list = [self.model_epoch_list[i] for i in similarity_model_idx_list]
        estimated_accuracy = self._estimate_model_accuracy(input_model_epoch, neighbor_model_epoch_list,
                                                           neighbor_model_accuracy_list)

        return estimated_accuracy

    @staticmethod
    def _sigmoid_function(x, x0, k):
        y = 1 / (1 + np.exp(-k * (x - x0)))
        return y

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
    def _build_conv_model_dataset(conv_model_list):
        conv_model_dataset = list()
        conv_model_accuracy_list = list()
        conv_model_epoch_list = list()
        for model_info in conv_model_list:
            conv_model_info_dict = dict()
            conv_model_info_list = model_info['model_name'].split('-')

            conv_model_info_dict['input_size'] = int(conv_model_info_list[0])
            conv_model_info_dict['channel_num'] = int(conv_model_info_list[1])
            conv_model_info_dict['class_num'] = int(conv_model_info_list[2])
            conv_model_info_dict['batch_size'] = int(conv_model_info_list[3])
            conv_model_info_dict['conv_layer_num'] = int(conv_model_info_list[4])
            conv_model_info_dict['pooling_layer_num'] = int(conv_model_info_list[5])
            conv_model_info_dict['residual_layer_num'] = int(conv_model_info_list[6])
            conv_model_info_dict['learning_rate'] = float(conv_model_info_list[7])
            conv_model_info_dict['optimizer'] = conv_model_info_list[8]
            conv_model_info_dict['activation'] = conv_model_info_list[9]
            conv_model_dataset.append(conv_model_info_dict)

            conv_model_epoch_list.append(int(conv_model_info_list[10]))
            conv_model_accuracy_list.append(float(model_info['model_accuracy']))

        return conv_model_dataset, conv_model_accuracy_list, conv_model_epoch_list

    def _compute_model_similarity(self, center_model, candidate_models):
        assert len(center_model) == len(candidate_models[0]), 'the input model and the candidate models are not in the same format'

        curmodel_similarity_list = list()
        candidate_models_similarity_list = list()

        for model_idx in candidate_models:
            curmodel_hyperparam_similarity_list = list()
            input_binary_set = set()
            candidate_binary_set = set()
            for cfg_idx in center_model:
                if cfg_idx in ('channel_num', 'class_num', 'optimizer', 'activation'):
                    #input_size_similarity = 1 if center_model[cfg_idx] == model_idx[cfg_idx] else 0
                    #curmodel_hyperparam_similarity_list.append(input_size_similarity)
                    input_binary_set.add(center_model[cfg_idx])
                    candidate_binary_set.add(model_idx[cfg_idx])
                else:
                    max_x = max([center_model[cfg_idx], model_idx[cfg_idx]])
                    diff_x = np.abs(center_model[cfg_idx] - model_idx[cfg_idx])
                    input_size_similarity = 1 - diff_x / max_x
                    curmodel_hyperparam_similarity_list.append(input_size_similarity)

            curmodel_binary_similarity = self._jaccard_index(input_binary_set, candidate_binary_set)
            curmodel_similarity_list.append(curmodel_binary_similarity)
            curmodel_overall_similarity = sum(curmodel_hyperparam_similarity_list) / len(curmodel_hyperparam_similarity_list)

            candidate_models_similarity_list.append(curmodel_overall_similarity)

        return candidate_models_similarity_list

    def _rank_model_similarity(self, candidate_models_info_list, candidate_models_similarity_list):
        assert len(candidate_models_info_list) >= self.similarity_top_k, 'the number of expected selected models is larger than the candidate models'

        similarity_sorted_idx_list = sorted(range(len(candidate_models_similarity_list)),
                                            key=lambda k: candidate_models_similarity_list[k],
                                            reverse=True)[:self.similarity_top_k]

        return similarity_sorted_idx_list

    def _estimate_model_accuracy(self, input_model_epoch, predict_model_epoch_list, predict_model_accuracy_list):
        popt, pcov = curve_fit(self._sigmoid_function,
                               predict_model_epoch_list,
                               predict_model_accuracy_list,
                               method='dogbox')

        estimated_accuracy = self._sigmoid_function(input_model_epoch, popt[0], popt[1])
        return estimated_accuracy

'''
if __name__ == "__main__":
    INPUT_MODEL_INFO = '224-3-1000-64-1-1-5-0.001-SGD-relu-18'
    input_model_list = INPUT_MODEL_INFO.split('-')
    input_model_dict = dict()

    input_model_dict['input_size'] = int(input_model_list[0])
    input_model_dict['channel_num'] = int(input_model_list[1])
    input_model_dict['class_num'] = int(input_model_list[2])
    input_model_dict['batch_size'] = int(input_model_list[3])
    input_model_dict['conv_layer_num'] = int(input_model_list[4])
    input_model_dict['pooling_layer_num'] = int(input_model_list[5])
    input_model_dict['residual_layer_num'] = int(input_model_list[6])
    input_model_dict['learning_rate'] = float(input_model_list[7])
    input_model_dict['optimizer'] = input_model_list[8]
    input_model_dict['activation'] = input_model_list[9]

    input_model_epoch = int(input_model_list[10])

    AE = AccuracyEstimator(1)
    AE.import_accuracy_dataset('/home/ruiliu/Development/mtml-tf/ml-sched/accuracy_dataset.json')
'''