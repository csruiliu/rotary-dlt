import numpy as np


def generate_workload(job_num, model_type_set, batch_size_set, optimizer_set, learn_rate_set, activation_set, train_data):
    sampled_model_type_list = np.random.choice(model_type_set, job_num, replace=True)
    sampled_batch_size_list = np.random.choice(batch_size_set, job_num, replace=True)
    sampled_optimizer_list = np.random.choice(optimizer_set, job_num, replace=True)
    sampled_learning_rate_list = np.random.choice(learn_rate_set, job_num, replace=True)
    sampled_activation_list = np.random.choice(activation_set, job_num, replace=True)

    sch_workload = list()

    for i in range(job_num):
        sch_model_config_dict = dict()
        sch_model_config_dict['job_id'] = i
        sch_model_config_dict['model_type'] = sampled_model_type_list[i]
        sch_model_config_dict['batch_size'] = sampled_batch_size_list[i]
        sch_model_config_dict['optimizer'] = sampled_optimizer_list[i]
        sch_model_config_dict['learning_rate'] = sampled_learning_rate_list[i]
        sch_model_config_dict['activation'] = sampled_activation_list[i]
        sch_model_config_dict['train_dataset'] = train_data
        #sch_model_config_dict['cur_accuracy'] = 0
        #sch_model_config_dict['prev_accuracy'] = 0
        sch_workload.append(sch_model_config_dict)

    return sch_workload
