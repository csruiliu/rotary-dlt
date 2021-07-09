import yaml
import os

current_folder = os.path.abspath(os.path.dirname(__file__))

with open(current_folder+'/config_rotary.yml', 'r') as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

dlt_workload_cfg = cfg['dlt_workload']
dlt_workload_size = dlt_workload_cfg['workload_size']
dlt_residual_ratio = dlt_workload_cfg['residual_ratio']
dlt_mobile_ratio = dlt_workload_cfg['mobile_ratio']
dlt_lstm_ratio = dlt_workload_cfg['lstm_ratio']
dlt_bert_ratio = dlt_workload_cfg['bert_ratio']
dlt_others_ratio = dlt_workload_cfg['others_ratio']

objective_cfg = cfg['objective']
convergence_ratio = objective_cfg['convergence_ratio']
accuracy_ratio = objective_cfg['accuracy_ratio']
runtime_ratio = objective_cfg['runtime_ratio']

random_seed = cfg['random_seed']
num_gpu = cfg['num_gpu']
running_slot = cfg['running_slot']