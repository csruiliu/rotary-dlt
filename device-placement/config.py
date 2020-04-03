import yaml

with open("config.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

##########################################
# General Hyperparameters
##########################################

hyperparams_base_cfg = cfg['base_parameter']

img_width = hyperparams_base_cfg['img_width']
img_height = hyperparams_base_cfg['img_height']
num_channels = hyperparams_base_cfg['num_channel']
num_classes = hyperparams_base_cfg['num_class']
rand_seed = hyperparams_base_cfg['random_seed']
device_num = hyperparams_base_cfg['device_num']
total_epochs = hyperparams_base_cfg['total_epochs']
available_cpu_num = hyperparams_base_cfg['available_cpu_num']
available_gpu_num = hyperparams_base_cfg['available_gpu_num']


##########################################
# Parameters for Workload
##########################################

hyperparams_workload_cfg = cfg['device_placement_workload']

workload_model_type = hyperparams_workload_cfg['workload_model_type']
workload_model_num = hyperparams_workload_cfg['workload_model_num']
workload_activation = hyperparams_workload_cfg['activation']
workload_opt = hyperparams_workload_cfg['optimizer']
workload_batch_size = hyperparams_workload_cfg['batch_size']
workload_num_layer = hyperparams_workload_cfg['num_model_layer']
workload_learning_rate = hyperparams_workload_cfg['learning_rate']
use_raw_image = hyperparams_workload_cfg['use_raw_image']
measure_step = hyperparams_workload_cfg['measure_step']
#same_input = hyperparams_workload_cfg['same_input']
#record_marker = hyperparams_workload_cfg['record_marker']
#batch_padding = hyperparams_workload_cfg['batch_padding']
#use_cpu = hyperparams_workload_cfg['use_cpu']
#use_tb_timeline = hyperparams_pack_cfg['use_tb_timeline']


##############################################################
# Experiment for profiling overhead of training CPU/GPU
##############################################################

exp_cpu_gpu_workload = cfg['exp_cpu_gpu_workload']
cpu_model_type = exp_cpu_gpu_workload['cpu_model_type']
cpu_model_num = exp_cpu_gpu_workload['cpu_model_num']
cpu_batch_size = exp_cpu_gpu_workload['cpu_batch_size']
cpu_optimizer = exp_cpu_gpu_workload['cpu_optimizer']
cpu_learning_rate = exp_cpu_gpu_workload['cpu_learning_rate']
cpu_activation = exp_cpu_gpu_workload['cpu_activation']
gpu_model_type = exp_cpu_gpu_workload['gpu_model_type']
gpu_model_num = exp_cpu_gpu_workload['gpu_model_num']
gpu_batch_size = exp_cpu_gpu_workload['gpu_batch_size']
gpu_optimizer = exp_cpu_gpu_workload['gpu_optimizer']
gpu_learning_rate = exp_cpu_gpu_workload['gpu_learning_rate']
gpu_activation = exp_cpu_gpu_workload['gpu_activation']
exp_marker = exp_cpu_gpu_workload['record_marker']


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


##########################################
# Path
##########################################

#path_cfg = cfg['local_path']
#path_cfg = cfg['roscoe_path']
path_cfg = cfg['hp_path']

mnist_train_img_path = path_cfg['mnist_train_img_path']
mnist_train_label_path = path_cfg['mnist_train_label_path']
mnist_t10k_img_path = path_cfg['mnist_t10k_img_path']
mnist_t10k_label_path = path_cfg['mnist_t10k_label_path']
cifar_10_path = path_cfg['cifar_10_path']
imagenet_t1k_img_path = path_cfg['imagenet_t1k_img_path']
imagenet_t1k_label_path = path_cfg['imagenet_t1k_label_path']
imagenet_t1k_bin_path = path_cfg['imagenet_t1k_bin_path']
imagenet_t10k_img_path = path_cfg['imagenet_t10k_img_path']
imagenet_t10k_label_path = path_cfg['imagenet_t10k_label_path']
imagenet_t10k_bin_path = path_cfg['imagenet_t10k_bin_path']
profile_path = path_cfg['profile_path']
ckpt_path = path_cfg['ckpt_path']
