import numpy as np
import json
import operator


class DLEstimator:
    def __init__(self, topk=5, poly_deg=4):
        # the list for all models' accuracy in the knowledgebase
        self.acc_model_list = list()

        # top k for selecting neighbours
        self.top_k = topk

        # the polynomial degree of the accuracy-epoch curve
        self.deg = poly_deg

        # the dict for all jobs data
        # key: str(input_model_dict['id']) + '-' + input_model_dict['model']
        # value is a dict with all necessary info: flag_list, acc_list, epoch_list
        self.job_predict_dict = dict()

    def import_accuracy_dataset(self, dataset_path):
        with open(dataset_path) as json_file:
            for item in json.load(json_file):
                self.acc_model_list.append(item)

    def get_accuracy_dataset(self):
        return self.acc_model_list

    def get_predict_dict(self):
        return self.job_predict_dict

    def import_workload(self, ml_workload):
        for m in ml_workload:
            m_key = str(m['id']) + '-' + m['model']

            tmp_dict = dict()
            tmp_dict['flag'] = list()
            tmp_dict['accuracy'] = list()
            tmp_dict['epoch'] = list()

            self.job_predict_dict[m_key] = tmp_dict

    def prepare_workload(self, ml_workload):
        for m in ml_workload:
            m_key = str(m['id']) + '-' + m['model']
            job_predict_info = dict()
            accuracy_list = list()
            epoch_list = list()

            neighbour_model_acc_list = self.compute_model_similarity_acc(m, self.acc_model_list)

            for nb_model in neighbour_model_acc_list:
                for aidx, acc in enumerate(nb_model['accuracy']):
                    accuracy_list.append(acc)
                    epoch_list.append(aidx)

            flag_list = [0] * len(accuracy_list)

            job_predict_info['flag'] = flag_list
            job_predict_info['accuracy'] = accuracy_list
            job_predict_info['epoch'] = epoch_list

            self.job_predict_dict[m_key] = job_predict_info

    def predict_accuracy(self, input_model_dict, input_model_epoch):
        job_key = str(input_model_dict['id']) + '-' + input_model_dict['model']

        job_predict_info = self.job_predict_dict[job_key]
        accuracy_list = job_predict_info['accuracy']
        epoch_list = job_predict_info['epoch']
        # the list for historical and actual accuracy data, 1 is actual accuracy, 0 is historical data
        flag_list = job_predict_info['flag']

        # init a new curve weight list for this prediction
        curve_weight_list = list()

        actual_dp_num = flag_list.count(1)
        base_dp_num = len(flag_list) - actual_dp_num

        actual_weight = 1 / (actual_dp_num + 1)
        base_weight = actual_weight / base_dp_num

        for flag in flag_list:
            if flag:
                curve_weight_list.append(actual_weight)
            else:
                curve_weight_list.append(base_weight)

        coefs = np.polyfit(x=np.asarray(epoch_list), y=np.asarray(accuracy_list), deg=self.deg, w=curve_weight_list)

        acc_estimation = np.polyval(coefs, input_model_epoch)

        return acc_estimation, coefs

    def predict_epoch(self, input_model_dict, input_model_accuracy):
        job_key = str(input_model_dict['id']) + '-' + input_model_dict['model']

        job_predict_info = self.job_predict_dict[job_key]
        accuracy_list = job_predict_info['accuracy']
        epoch_list = job_predict_info['epoch']
        # the list for historical and actual accuracy data, 1 is actual accuracy, 0 is historical data
        flag_list = job_predict_info['flag']

        # init a new curve weight list for this prediction
        curve_weight_list = list()

        actual_dp_num = flag_list.count(1)
        base_dp_num = len(flag_list) - actual_dp_num

        actual_weight = 1 / (actual_dp_num + 1)
        base_weight = actual_weight / base_dp_num

        for flag in flag_list:
            if flag:
                curve_weight_list.append(actual_weight)
            else:
                curve_weight_list.append(base_weight)

        coefs = np.polyfit(x=np.asarray(accuracy_list), y=np.asarray(epoch_list), deg=self.deg, w=curve_weight_list)

        epoch_estimation = np.polyval(coefs, input_model_accuracy)

        return epoch_estimation, coefs

    def add_actual_data(self, job_key, accuracy, epoch):
        job_predict_info = self.job_predict_dict[job_key]
        job_predict_info['flag'].append(1)
        job_predict_info['accuracy'].append(accuracy)
        job_predict_info['epoch'].append(epoch)

    def compute_model_similarity_acc(self, center_model, candidate_models):
        similarity_list = list()

        for candidate in candidate_models:
            ''' 
                We only take the candidate model that has: 
                1. same training dataset 
                2. same output class 
                3. same learning rate
                4. same optimizer 
                Otherwise, set the similarity as -1
            '''
            if (center_model['training_data'] == candidate['training_data'] and
                # center_model['classes'] == candidate['classes'] and
                center_model['learn_rate'] == candidate['learn_rate'] and
                center_model['batch_size'] == candidate['batch_size']):

                max_x = max([center_model['num_parameters'], candidate['num_parameters']])
                diff_x = np.abs(center_model['num_parameters'] - candidate['num_parameters'])
                parameter_similarity = 1 - diff_x / max_x
                similarity_list.append(parameter_similarity)
            else:
                similarity_list.append(-1)

        ''' get the top k models from mlbase according to the similarity '''
        similarity_sorted_idx_list = sorted(range(len(similarity_list)),
                                            key=lambda k: similarity_list[k],
                                            reverse=True)[:self.top_k]

        topk_model_list = operator.itemgetter(*similarity_sorted_idx_list)(self.acc_model_list)

        return topk_model_list
