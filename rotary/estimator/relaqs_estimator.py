import json
import numpy as np
from scipy.optimize import curve_fit


class ReLAQSEstimator:
    def __init__(self, batch_size=32, num_worker=1):
        self._knowledge_list = list()
        self._job_predict_dict = dict()
        self.batch_size = batch_size
        self.num_worker = num_worker

    def import_knowledge_archive(self, knowledge_archive):
        with open(knowledge_archive) as ka:
            for knowledge_item in json.load(ka):
                self._knowledge_list.append(knowledge_item)

    def func_progress(self, x, A, B):
        return 1 / (A * x*x + B)

    def func_runtime(self, x, A, B):
        return A * (x * self.batch_size / self.num_worker) + B

    def fit_progress(self, x, y):
        opt, cov = curve_fit(self.func_progress, x, y)
        return opt, cov

    def fit_runtime(self, x, y):
        opt, cov = curve_fit(self.func_runtime, x, y)
        return opt, cov

    def predict(self, input_model_dict, input_x, mode):
        job_key = str(input_model_dict['id']) + '-' + input_model_dict['model']

        if job_key in self._job_predict_dict:
            # if the job_key exists, get the predict info
            job_predict_info = self._job_predict_dict[job_key]
            accuracy_list = job_predict_info['accuracy']
            epoch_list = job_predict_info['epoch']
        else:
            # if this is a new model for prediction, compute and generate the predict info
            job_predict_info = dict()
            accuracy_list = list()
            epoch_list = list()

            for nb_model in self._knowledge_list:
                for aidx, acc in enumerate(nb_model['accuracy']):
                    accuracy_list.append(acc)
                    epoch_list.append(aidx)

            job_predict_info['accuracy'] = accuracy_list
            job_predict_info['epoch'] = epoch_list

            self._job_predict_dict[job_key] = job_predict_info

        if mode == 'accuracy':
            popt, pcov = self.fit_progress(np.asarray(epoch_list), np.asarray(accuracy_list))

            A = popt[0]
            B = popt[1]

            acc_estimation = self.func_progress(input_x, A, B)

            return acc_estimation

        elif mode == 'epoch':
            popt, pcov = self.fit_runtime(np.asarray(accuracy_list), np.asarray(epoch_list))

            A = popt[0]
            B = popt[1]

            epoch_estimation = self.func_runtime(input_x, A, B)

            return epoch_estimation

        else:
            raise ValueError('Predication ')
