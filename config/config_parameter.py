import yaml
import os

current_folder = os.path.abspath(os.path.dirname(__file__))

with open(current_folder+'/config_parameter.yml', 'r') as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)


##########################################
# Hyperparameters for input
##########################################

hyperparams_input = cfg['hyperparams_input']
img_width_imagenet = hyperparams_input['img_width_imagenet']
img_height_imagenet = hyperparams_input['img_height_imagenet']
img_width_cifar10 = hyperparams_input['img_width_cifar10']
img_height_cifar10 = hyperparams_input['img_height_cifar10']
img_width_mnist = hyperparams_input['img_width_mnist']
img_height_mnist = hyperparams_input['img_height_mnist']
num_class_imagenet = hyperparams_input['num_class_imagenet']
num_class_cifar10 = hyperparams_input['num_class_cifar10']
num_class_mnist = hyperparams_input['num_class_mnist']
num_channels_rgb = hyperparams_input['num_channel_rgb']
num_channels_bw = hyperparams_input['num_channel_bw']


##################################################
# General hyperparameters for Scheduling
##################################################

schedule_cfg = cfg['dlt_schedule_para']
sch_gpu_num = schedule_cfg['gpu_device_num']
sch_cpu_num = schedule_cfg['cpu_device_num']
sch_time_slots_num = schedule_cfg['time_slots_num']
sch_slot_time_period = schedule_cfg['slot_time_period']


##################################################
# Scheduling workload for SLO
##################################################

schedule_workload_slo_cfg = cfg['dlt_schedule_workload_slo']
slo_job_num = schedule_workload_slo_cfg['job_num']
train_dataset = schedule_workload_slo_cfg['train_dataset']
slo_model_type_set = schedule_workload_slo_cfg['model_type_set']
slo_batch_size_set = schedule_workload_slo_cfg['batch_size_set']
slo_optimizer_set = schedule_workload_slo_cfg['optimizer_set']
slo_learning_rate_set = schedule_workload_slo_cfg['learning_rate_set']
slo_activation_set = schedule_workload_slo_cfg['activation_set']
slo_reward_function = schedule_workload_slo_cfg['reward_function']


##################################################
# Scheduling workload for hyperparameter search
##################################################

schedule_workload_hpsearch_cfg = cfg['dlt_schedule_workload_hpsearch']
hpsearch_job_num = schedule_workload_hpsearch_cfg['job_num']
hpsearch_train_dataset = schedule_workload_hpsearch_cfg['train_dataset']
hpsearch_model_type = schedule_workload_hpsearch_cfg['model_type']
hpsearch_layer_set = schedule_workload_hpsearch_cfg['layer_set']
hpsearch_batch_size_set = schedule_workload_hpsearch_cfg['batch_size_set']
hpsearch_optimizer_set = schedule_workload_hpsearch_cfg['optimizer_set']
hpsearch_learning_rate_set = schedule_workload_hpsearch_cfg['learning_rate_set']


##############################################################
# Experiment for profiling overhead of training CPU/GPU
##############################################################

profile_exp_cpu_gpu = cfg['profile_concur_cpugpu_workload']
cpu_model_type = profile_exp_cpu_gpu['cpu_model_type']
cpu_model_num = profile_exp_cpu_gpu['cpu_model_num']
cpu_batch_size = profile_exp_cpu_gpu['cpu_batch_size']
cpu_optimizer = profile_exp_cpu_gpu['cpu_optimizer']
cpu_learning_rate = profile_exp_cpu_gpu['cpu_learning_rate']
cpu_activation = profile_exp_cpu_gpu['cpu_activation']
gpu_model_type = profile_exp_cpu_gpu['gpu_model_type']
gpu_model_num = profile_exp_cpu_gpu['gpu_model_num']
gpu_batch_size = profile_exp_cpu_gpu['gpu_batch_size']
gpu_optimizer = profile_exp_cpu_gpu['gpu_optimizer']
gpu_learning_rate = profile_exp_cpu_gpu['gpu_learning_rate']
gpu_activation = profile_exp_cpu_gpu['gpu_activation']


