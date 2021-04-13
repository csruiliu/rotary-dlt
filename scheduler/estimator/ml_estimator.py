import numpy as np
import json
import matplotlib.pyplot as plt


class MLEstimator:
    def __init__(self, input_model_dict, top_k):
        self.top_k = top_k
        self.input_model_dict = input_model_dict

        # list stores model info with accuracy of various epochs
        self.acc_model_list = None

        # list stores model info with training steptime
        self.steptime_model_list = None

        # lists for predicting accuracy for current model
        self.acc_list = None
        self.acc_epoch_list = None
        self.acc_weight_list = None
        self.acc_flag_list = None

        # lists for predicting steptime for current model
        self.steptime_list = None
        self.steptime_weight_list = None
        self.steptime_flag_list = None

    def import_accuracy_dataset(self, dataset_path):
        with open(dataset_path) as json_file:
            self.acc_model_list = json.load(json_file)

        self.acc_list = list()
        self.acc_epoch_list = list()
        self.acc_flag_list = list()

        neighbour_model_acc_list = self.compute_model_similarity_acc(self.input_model_dict, self.acc_model_list)

        for model in neighbour_model_acc_list:
            for aidx, acc in enumerate(model['accuracy']):
                self.acc_list.append(acc)
                self.acc_epoch_list.append(aidx)

        self.acc_flag_list = [0] * len(self.acc_list)

    def import_steptime_dataset(self, dataset_path):
        with open(dataset_path) as json_file:
            self.steptime_model_list = json.load(json_file)

        self.steptime_list = list()

        for st in self.steptime_model_list:
            self.steptime_list.append(st['steptime'])

    def compute_model_similarity_acc(self, center_model, candidate_models):
        ''' similarity between each model in mlbase and the center model '''
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
                center_model['classes'] == candidate['classes'] and
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

        topk_model_list = [self.acc_model_list[i] for i in similarity_sorted_idx_list]

        return topk_model_list

    def distill_actual_data(self, actual_data):
        self.acc_epoch_list.append(actual_data[0])
        self.acc_list.append(actual_data[1])
        self.acc_flag_list.append(1)

    def predict_accuracy(self, input_model_epoch):
        self.weight_list = list()
        actual_dp_num = self.acc_flag_list.count(1)
        base_dp_num = len(self.acc_flag_list) - actual_dp_num

        actual_weight = 1 / (actual_dp_num + 1)
        base_weight = actual_weight / base_dp_num

        for flag in self.acc_flag_list:
            if flag:
                self.weight_list.append(actual_weight)
            else:
                self.weight_list.append(base_weight)

        coefs = np.polyfit(x=np.asarray(self.acc_epoch_list), y=np.asarray(self.acc_list), deg=3, w=self.weight_list)

        plt.figure()
        plt.plot(np.arange(1, 21), np.polyval(coefs, np.arange(1, 21)), color="black")
        plt.plot(np.arange(1, 21), [0.10569999970495701,
                                    0.17059999911114573,
                                    0.23050000064074994,
                                    0.30139999993145467,
                                    0.3693000010401011,
                                    0.4184000000357628,
                                    0.4499999986588955,
                                    0.505199998319149,
                                    0.5318000014126301,
                                    0.5642000022530556,
                                    0.5793000020086765,
                                    0.5967000035941601,
                                    0.6219000032544136,
                                    0.6229000036418438,
                                    0.6461000037193299,
                                    0.6477000056207181,
                                    0.6603000056743622,
                                    0.6548000074923038,
                                    0.6700000047683716,
                                    0.6700000068545342], color="black")
        plt.xticks(np.arange(1, 21))
        plt.show()

        acc_estimation = np.polyval(coefs, input_model_epoch)

        return acc_estimation


if __name__ == "__main__":
    input_model = dict()

    input_model['model_name'] = 'model_demo'
    input_model['num_parameters'] = 2258538
    input_model['batch_size'] = 128
    input_model['opt'] = 'Momentum'
    input_model['learn_rate'] = 0.01
    input_model['training_data'] = 'cifar'
    input_model['classes'] = 10

    ml_estimator = MLEstimator(input_model, top_k=5)
    ml_estimator.import_accuracy_dataset('/home/ruiliu/Development/ml-estimator/mlbase/model_acc.json')
    ml_estimator.distill_actual_data([1, 0.10569999970495701])
    # ml_estimator.distill_actual_data([2, 0.17059999911114573])
    ml_estimator.distill_actual_data([3, 0.23050000064074994])
    # ml_estimator.distill_actual_data([4, 0.30139999993145467])
    ml_estimator.distill_actual_data([5, 0.3693000010401011])
    # ml_estimator.distill_actual_data([6, 0.4184000000357628])
    ml_estimator.distill_actual_data([7, 0.4499999986588955])
    acc = ml_estimator.predict_accuracy(10)

    print(acc)
