import numpy as np


def generate_workload(job_num, model_type_set, batch_size_set, optimizer_set, learn_rate_set, activation_set,
                      train_data, use_seed=False):

    if use_seed:
        np.random.seed(999)

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
        if sch_model_config_dict['model_type'] == 'resnet':
            sch_model_config_dict['model_layer_num'] = np.random.choice([18, 34, 50, 101, 152], replace=False)
        elif sch_model_config_dict['model_type'] == 'densenet':
            sch_model_config_dict['model_layer_num'] = np.random.choice([121, 169, 201, 264], replace=False)
        else:
            sch_model_config_dict['model_layer_num'] = np.random.randint(1, 10)
        sch_model_config_dict['batch_size'] = sampled_batch_size_list[i]
        sch_model_config_dict['optimizer'] = sampled_optimizer_list[i]
        sch_model_config_dict['learning_rate'] = sampled_learning_rate_list[i]
        sch_model_config_dict['activation'] = sampled_activation_list[i]
        sch_model_config_dict['train_dataset'] = train_data
        sch_workload.append(sch_model_config_dict)

    return sch_workload
