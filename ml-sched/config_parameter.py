import yaml

with open("config_parameter.yml", 'r') as ymlfile:
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

##########################################
# Hyperparameters for measurement
##########################################

hyperparams_measure_cfg = cfg['hyperparams_measure']

rand_seed = hyperparams_measure_cfg['random_seed']
batch_padding = hyperparams_measure_cfg['batch_padding']
record_marker = hyperparams_measure_cfg['record_marker']
use_cpu = hyperparams_measure_cfg['use_cpu']
use_raw_image = hyperparams_measure_cfg['use_raw_image']
measure_step = hyperparams_measure_cfg['measure_step']
use_tb_timeline = hyperparams_measure_cfg['use_tb_timeline']
same_input = hyperparams_measure_cfg['same_input']
available_cpu_num = hyperparams_measure_cfg['available_cpu_num']
available_gpu_num = hyperparams_measure_cfg['available_gpu_num']

##################################################
# Hyperparameters placement schedule
##################################################

placement_schedule_workload_cfg = cfg['placement_schedule']
sch_gpu_num = placement_schedule_workload_cfg['gpu_device_num']
sch_cpu_num = placement_schedule_workload_cfg['cpu_device_num']
sch_job_num = placement_schedule_workload_cfg['job_num']
sch_time_slots_num = placement_schedule_workload_cfg['time_slots_num']
sch_slot_time_period = placement_schedule_workload_cfg['slot_time_period']
sch_model_type_set = placement_schedule_workload_cfg['model_type_set']
sch_batch_size_set = placement_schedule_workload_cfg['batch_size_set']
sch_optimizer_set = placement_schedule_workload_cfg['optimizer_set']
sch_learning_rate_set = placement_schedule_workload_cfg['learning_rate_set']
sch_activation_set = placement_schedule_workload_cfg['activation_set']
sch_reward_function = placement_schedule_workload_cfg['reward_function']
placement_proportion_rate = placement_schedule_workload_cfg['placement_proportion_rate']


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

##########################################
# Simple device placement
##########################################

simple_placement_cfg = cfg['simple_placement']
simple_placement_init_res = simple_placement_cfg['init_resource_conf']
simple_placement_up_rate = simple_placement_cfg['up_rate']
simple_placement_discard_rate = simple_placement_cfg['discard_rate']

##########################################
# Robin device placement
##########################################

robin_device_placement = cfg['robin_placement']
robin_time_limit = robin_device_placement['time_limit']

