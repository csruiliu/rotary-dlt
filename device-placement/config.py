import yaml

with open("config.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

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

##########################################
# Parameters for Workload
##########################################

#hyperparams_workload_cfg = cfg['device_placement_workload']
#workload_model_type = hyperparams_workload_cfg['workload_model_type']
#workload_model_num = hyperparams_workload_cfg['workload_model_num']
#workload_activation = hyperparams_workload_cfg['activation']
#workload_opt = hyperparams_workload_cfg['optimizer']
#workload_batch_size = hyperparams_workload_cfg['batch_size']
#workload_num_layer = hyperparams_workload_cfg['num_model_layer']
#workload_learning_rate = hyperparams_workload_cfg['learning_rate']

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
