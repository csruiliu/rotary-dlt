import yaml
import os

current_folder = os.path.abspath(os.path.dirname(__file__))

with open(current_folder+'/config_rotary.yml', 'r') as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

dlt_workload_cfg = cfg['dlt_workload']
dlt_workload_size = dlt_workload_cfg['workload_size']
dlt_cv_light_ratio = dlt_workload_cfg['cv_light_ratio']
dlt_cv_med_ratio = dlt_workload_cfg['cv_med_ratio']
dlt_cv_heavy_ratio = dlt_workload_cfg['cv_heavy_ratio']
dlt_nlp_light_ratio = dlt_workload_cfg['nlp_light_ratio']
dlt_nlp_med_ratio = dlt_workload_cfg['nlp_med_ratio']
dlt_nlp_heavy_ratio = dlt_workload_cfg['nlp_heavy_ratio']

cv_workload_cfg = cfg['cv_workload']
cv_workload_size = cv_workload_cfg['workload_size']
cv_light_ratio = cv_workload_cfg['light_ratio']
cv_med_ratio = cv_workload_cfg['med_ratio']
cv_heavy_ratio = cv_workload_cfg['heavy_ratio']

nlp_workload_cfg = cfg['nlp_workload']
nlp_workload_size = nlp_workload_cfg['workload_size']
nlp_light_ratio = nlp_workload_cfg['light_ratio']
nlp_med_ratio = nlp_workload_cfg['med_ratio']
nlp_heavy_ratio = nlp_workload_cfg['heavy_ratio']

objective_cfg = cfg['objective']
convergence_ratio = objective_cfg['convergence_ratio']
accuracy_ratio = objective_cfg['accuracy_ratio']
runtime_ratio = objective_cfg['runtime_ratio']

random_seed = cfg['random_seed']
num_gpu = cfg['num_gpu']
running_slot = cfg['running_slot']