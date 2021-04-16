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
                 random_seed):

        # the light, medium, heavy due to the number of parameters
        self._workload_size = workload_size
        self._cv_light_ratio = cv_light_ratio
        self._cv_med_ratio = cv_med_ratio
        self._cv_heavy_ratio = cv_heavy_ratio
        self._nlp_light_ratio = nlp_light_ratio
        self._nlp_med_ratio = nlp_med_ratio
        self._nlp_heavy_ratio = nlp_heavy_ratio

        self._cv_model_light = ['inception', 'mobilenet', 'mobilenet_v2', 'squeezenet', 'xception']
        self._cv_model_med = ['shufflenet', 'shufflenet_v2', 'resnet', 'resnext', 'efficientnet']
        self._cv_model_heavy = ['lenet', 'vgg', 'alexnet', 'zfnet', 'densenet']

        self._nlp_model_light = ['nnlm', 'word2vec']
        self._nlp_model_med = ['textrnn', 'bilstm']
        self._nlp_model_heavy = ['seq2seq', 'transformer']

        self._batch_size_list = [25, 32, 50, 64, 100, 128]
        self._epoch_list = [1, 5, 10, 100, 500]

        self._random_seed = random_seed

    @staticmethod
    def _model_selection(model_list, model_num):
        return list(np.random.choice(model_list, size=model_num))

    def generate_workload(self):
        np.random.seed(self._random_seed)

        cv_light_num = int(self._cv_light_ratio * self._workload_size)
        cv_med_num = int(self._cv_med_ratio * self._workload_size)
        cv_heavy_num = int(self._cv_light_ratio * self._workload_size)
        nlp_light_num = int(self._nlp_light_ratio * self._workload_size)
        nlp_med_num = int(self._nlp_med_ratio * self._workload_size)
        nlp_heavy_num = int(self._nlp_heavy_ratio * self._workload_size)

        cv_light_list = self._model_selection(self._cv_model_light, cv_light_num)
        cv_med_list = self._model_selection(self._cv_model_med, cv_med_num)
        cv_heavy_list = self._model_selection(self._cv_model_heavy, cv_heavy_num)
        nlp_light_list = self._model_selection(self._nlp_model_light, nlp_light_num)
        nlp_med_list = self._model_selection(self._nlp_model_med, nlp_med_num)
        nlp_heavy_list = self._model_selection(self._nlp_model_heavy, nlp_heavy_num)

        model_selection_list = cv_light_list + cv_med_list + cv_heavy_list + nlp_light_list + nlp_med_list + nlp_heavy_list
        np.random.shuffle(model_selection_list)

        workload = list()
        for midx, model in enumerate(model_selection_list):
            job = list()
            job.append(model)
            job.append(np.random.choice(self._batch_size_list, size=1)[0])
            job.append(np.random.choice(self._epoch_list, size=1)[0])
            workload.append(job)
        return workload


if __name__ == "__main__":
    workload_size = 20
    cv_light_ratio = 0.4
    cv_med_ratio = 0.2
    cv_heavy_ratio = 0.2
    nlp_light_ratio = 0.3
    nlp_med_ratio = 0.3
    nlp_heavy_ratio = 0.4
    random_seed = 10000

    wg = WorkloadGenerator(workload_size,
                           cv_light_ratio,
                           cv_med_ratio,
                           cv_heavy_ratio,
                           nlp_light_ratio,
                           nlp_med_ratio,
                           nlp_heavy_ratio,
                           random_seed)

    dl_workload = wg.generate_workload()
    print(dl_workload)

