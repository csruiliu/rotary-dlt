import numpy as np
import json
import operator


class RotaryEstimator:
    def __init__(self, topk=5, poly_deg=3):
        # the list for storing the training information for progress in the knowledgebase
        self._progress_knowledge_list = list()

        # the list for storing the training information for memory in the knowledgebase
        self._memory_knowledge_list = list()

        # top k for selecting neighbours
        self._top_k = topk

        # the polynomial degree of the accuracy-epoch curve
        self._deg = poly_deg

        # the dict for all jobs data
        # key: str(input_model_dict['id']) + '-' + input_model_dict['model']
        # value is a dict with all necessary info: flag_list, acc_list, epoch_list
        self._job_progress_predict_dict = dict()

        # the dict for all jobs data
        # key: str(input_model_dict['id']) + '-' + input_model_dict['model']
        # value is a predicted memory
        self._job_memory_predict_dict = dict()

    def _compute_similarity_progress(self, center_model, candidate_models):
        similarity_list = list()

        for candidate in candidate_models:
            ''' 
                We only take the candidate model that has: 
                1. same training dataset 
                2. same learning rate 
                3. same batch size
                4. same optimizer
                Otherwise, set the similarity as -1
            '''
            if (center_model['training_data'] == candidate['training_data'] and
                center_model['learn_rate'] == candidate['learn_rate'] and
                center_model['opt'] == candidate['opt'] and
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
                                            reverse=True)[:self._top_k]

        topk_model_list = operator.itemgetter(*similarity_sorted_idx_list)(self._progress_knowledge_list)

        return topk_model_list

    def _compute_similarity_memory(self, center_model, candidate_models):
        similarity_list = list()
        for candidate in candidate_models:
            max_x = max([center_model['num_parameters'], candidate['num_parameters']])
            diff_x = np.abs(center_model['num_parameters'] - candidate['num_parameters'])
            parameter_similarity = 1 - diff_x / max_x

            similarity_list.append(parameter_similarity)
        else:
            similarity_list.append(-1)

        similarity_sorted_idx_list = sorted(range(len(similarity_list)),
                                            key=lambda k: similarity_list[k],
                                            reverse=True)[:self._top_k]

        topk_model_list = operator.itemgetter(*similarity_sorted_idx_list)(self._progress_knowledge_list)

        return topk_model_list

    def import_progress_knowledge_archive(self, knowledge_archive):
        with open(knowledge_archive) as ka:
            for knowledge_item in json.load(ka):
                self._progress_knowledge_list.append(knowledge_item)

    def import_progress_knowledge_realtime(self, job_key, accuracy, epoch):
        job_predict_info = self._job_progress_predict_dict[job_key]
        job_predict_info['flag'].append(1)
        job_predict_info['accuracy'].append(accuracy)
        job_predict_info['epoch'].append(epoch)

    def import_memory_knowledge_archive(self, knowledge_archive):
        with open(knowledge_archive) as ka:
            for knowledge_item in json.load(ka):
                self._memory_knowledge_list.append(knowledge_item)

    def predict(self, input_model_dict, input_x, mode):
        """
            :parameter
            input_model_dict: the architecture and hyperparameters of the input model
            input_x: the x of a linear model for prediction, e.g., accuracy or epoch
            mode: predicting 'accuracy' or 'epoch'

            if predicting accuracy, input_x is $epoch, mode='accuracy'.
            if predicting epoch, input_x is $accuracy, mode='epoch'.
        """
        job_key = str(input_model_dict['id']) + '-' + input_model_dict['model']

        if job_key in self._job_progress_predict_dict:
            # if the job_key exists, get the predict info
            job_predict_info = self._job_progress_predict_dict[job_key]
            accuracy_list = job_predict_info['accuracy']
            epoch_list = job_predict_info['epoch']
        else:
            # if this is a new model for prediction, compute and generate the predict info
            job_predict_info = dict()
            accuracy_list = list()
            epoch_list = list()

            neighbour_model_acc_list = self._compute_similarity_progress(input_model_dict, self._progress_knowledge_list)

            for nb_model in neighbour_model_acc_list:
                for aidx, acc in enumerate(nb_model['accuracy']):
                    accuracy_list.append(acc)
                    epoch_list.append(aidx)

            flag_list = [0] * len(accuracy_list)

            job_predict_info['flag'] = flag_list
            job_predict_info['accuracy'] = accuracy_list
            job_predict_info['epoch'] = epoch_list

            self._job_progress_predict_dict[job_key] = job_predict_info

        # the list for historical and actual accuracy data, 1 is realtime accuracy, 0 is archive data
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

        if mode == 'accuracy':
            coefs = np.polyfit(x=np.asarray(epoch_list),
                               y=np.asarray(accuracy_list),
                               deg=self._deg,
                               w=curve_weight_list)

            acc_estimation = np.polyval(coefs, input_x)

            return acc_estimation

        elif mode == 'epoch':
            coefs = np.polyfit(x=np.asarray(accuracy_list),
                               y=np.asarray(epoch_list),
                               deg=self._deg,
                               w=curve_weight_list)

            epoch_estimation = np.polyval(coefs, input_x)

            return epoch_estimation

        else:
            raise ValueError('Predication ')

    def predict_memory_consumption(self, input_model_dict, input_x):
        """
            :parameter
            input_model_dict: the architecture and hyperparameters of the input model
            input_x: the x of a linear model for prediction, e.g., batch_size
        """
        job_key = str(input_model_dict['id']) + '-' + input_model_dict['model']

        if job_key in self._job_memory_predict_dict:
            max_memory = self._job_memory_predict_dict[job_key]
            return max_memory
        else:
            neighbour_model_acc_list = self._compute_similarity_progress(input_model_dict, self._progress_knowledge_list)

        model_weight = 1
        model_weight_list = list()
        batch_size_list = list()
        memory_list = list()

        for item_idx, model in enumerate(neighbour_model_acc_list):
            if item_idx == len(neighbour_model_acc_list) - 1:
                model_weight_list.append(model_weight)
            else:
                model_weight_list.append(model_weight / 2)

            model_weight = model_weight/2
            batch_size_list.append(model['batch_size'])
            memory_list.append(model['memory'])

        coefs = np.polyfit(x=np.asarray(batch_size_list),
                           y=np.asarray(memory_list),
                           deg=self._deg,
                           w=model_weight_list)

        memory_estimation = np.polyval(coefs, input_x)
        self._job_memory_predict_dict[job_key] = memory_estimation

        return memory_estimation

    def get_progress_knowledge_list(self):
        return self._progress_knowledge_list
