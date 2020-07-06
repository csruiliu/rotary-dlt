import numpy as np
from scipy.optimize import curve_fit

import model_accuracy_config as acc_json_cfg


def sigmoid_function(x, x0, k):
    y = 1 / (1 + np.exp(-k * (x - x0)))
    return y


def jaccard_index(s1, s2):
    size_s1 = len(s1)
    size_s2 = len(s2)

    # Get the intersection set
    intersect = s1 & s2
    size_in = len(intersect)

    # Calculate the Jaccard index using the formula
    jaccard_idx = size_in / (size_s1 + size_s2 - size_in)

    return jaccard_idx


def generate_conv_model_dataset():
    conv_model_dataset = list()
    conv_model_accuracy_list = list()
    conv_model_epoch_list = list()
    for model_info in acc_json_cfg.conv_model_list:
        conv_model_info_dict = dict()
        conv_model_info_list = model_info['model_name'].split('-')

        conv_model_info_dict['input_size'] = int(conv_model_info_list[0])
        conv_model_info_dict['channel_num'] = int(conv_model_info_list[1])
        conv_model_info_dict['class_num'] = int(conv_model_info_list[2])
        conv_model_info_dict['batch_size'] = int(conv_model_info_list[3])
        conv_model_info_dict['conv_layer_num'] = int(conv_model_info_list[4])
        conv_model_info_dict['pooling_layer_num'] = int(conv_model_info_list[5])
        conv_model_info_dict['total_layer_num'] = int(conv_model_info_list[6])
        conv_model_info_dict['learning_rate'] = float(conv_model_info_list[7])
        conv_model_info_dict['optimizer'] = conv_model_info_list[8]
        conv_model_info_dict['activation'] = conv_model_info_list[9]
        conv_model_dataset.append(conv_model_info_dict)

        conv_model_epoch_list.append(int(conv_model_info_list[10]))
        conv_model_accuracy_list.append(float(model_info['model_accuracy']))

    return conv_model_dataset, conv_model_accuracy_list, conv_model_epoch_list


def compute_model_similarity(center_model, candidate_models):
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

        curmodel_binary_similarity = jaccard_index(input_binary_set, candidate_binary_set)
        curmodel_similarity_list.append(curmodel_binary_similarity)
        curmodel_overall_similarity = sum(curmodel_hyperparam_similarity_list) / len(curmodel_hyperparam_similarity_list)

        candidate_models_similarity_list.append(curmodel_overall_similarity)

    return candidate_models_similarity_list


def rank_model_similarity(candidate_models_info_list, candidate_models_similarity_list, k=2):
    assert len(candidate_models_info_list) >= k, 'the number of expected selected models is larger than the candidate models'

    similarity_sorted_idx_list = sorted(range(len(candidate_models_similarity_list)),
                                        key=lambda k: candidate_models_similarity_list[k],
                                        reverse=True)[:k]

    return similarity_sorted_idx_list


def estimate_model_accuracy(input_model_epoch, predict_model_epoch_list, predict_model_accuracy_list):

    popt, pcov = curve_fit(sigmoid_function,
                           predict_model_epoch_list,
                           predict_model_accuracy_list,
                           method='dogbox')

    estimated_accuracy = sigmoid_function(input_model_epoch, popt[0], popt[1])
    return estimated_accuracy


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
    input_model_dict['total_layer_num'] = int(input_model_list[6])
    input_model_dict['learning_rate'] = float(input_model_list[7])
    input_model_dict['optimizer'] = input_model_list[8]
    input_model_dict['activation'] = input_model_list[9]

    input_model_epoch = int(input_model_list[10])

    model_info_list, model_accuracy_list, model_epoch_list = generate_conv_model_dataset()
    model_similarity_list = compute_model_similarity(input_model_dict, model_info_list)
    similarity_model_idx_list = rank_model_similarity(model_info_list, model_similarity_list)

    # return the nearest models info according to similarity
    # neighbor_model_info_list = [model_info_list[i] for i in similarity_model_idx_list]
    neighbor_model_accuracy_list = [model_accuracy_list[i] for i in similarity_model_idx_list]
    neighbor_model_epoch_list = [model_epoch_list[i] for i in similarity_model_idx_list]
    estimated_accuracy = estimate_model_accuracy(input_model_epoch, neighbor_model_epoch_list, neighbor_model_accuracy_list)

    print(estimated_accuracy)
