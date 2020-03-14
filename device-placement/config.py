import yaml

with open("config.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

##########################################
# Hyperparameters for pack training
##########################################

hyperparams_pack_cfg = cfg['auto_device_placement']

pack_img_width = hyperparams_pack_cfg['img_width']
pack_img_height = hyperparams_pack_cfg['img_height']
pack_num_channels = hyperparams_pack_cfg['num_channel']
pack_num_classes = hyperparams_pack_cfg['num_class']
pack_num_epoch = hyperparams_pack_cfg['num_epoch']
pack_rand_seed = hyperparams_pack_cfg['random_seed']
pack_model_type = hyperparams_pack_cfg['workload_model_type']
pack_activation = hyperparams_pack_cfg['activation']
pack_opt = hyperparams_pack_cfg['optimizer']
pack_batch_size = hyperparams_pack_cfg['batch_size']
pack_num_layer = hyperparams_pack_cfg['num_model_layer']
pack_learning_rate = hyperparams_pack_cfg['learning_rate']
pack_record_marker = hyperparams_pack_cfg['record_marker']
pack_batch_padding = hyperparams_pack_cfg['batch_padding']
pack_use_cpu = hyperparams_pack_cfg['use_cpu']
pack_use_raw_image = hyperparams_pack_cfg['use_raw_image']
pack_measure_step = hyperparams_pack_cfg['measure_step']
pack_same_input = hyperparams_pack_cfg['same_input']
pack_use_tb_timeline = hyperparams_pack_cfg['use_tb_timeline']

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
