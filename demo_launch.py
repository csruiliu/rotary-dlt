import argparse

from rotary.common.property_utils import PropertyUtils
from rotary.common.workload_generator import WorkloadGenerator
from rotary.sched.rotary import Rotary
from rotary.sched.srf import SRF
from rotary.sched.bcf import BCF
from rotary.sched.laf import LAF


def main():
    ###################################
    # get parameters from cli
    ###################################

    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--schedule', action='store', type=str,
                        choices=['laf', 'bcf', 'srf', 'rotary'],
                        required=True,
                        help='indicate schedule mechanism')

    args = parser.parse_args()
    sched_name = args.schedule

    #######################################################
    # get parameters from configuration
    #######################################################

    path_file = '/home/ruiliu/Develop/rotary-dlt/config/local_path_file.json'
    para_file = '/home/ruiliu/Develop/rotary-dlt/config/para_file.json'
    knowledgebase_folder = '/home/ruiliu/Develop/rotary-dlt/knowledgebase'

    para_cfg = PropertyUtils.load_property_file(properties_file=para_file)

    dlt_workload_cfg = para_cfg['dlt_workload']
    dlt_workload_size = dlt_workload_cfg['workload_size']
    dlt_residual_ratio = dlt_workload_cfg['residual_ratio']
    dlt_mobile_ratio = dlt_workload_cfg['mobile_ratio']
    dlt_lstm_ratio = dlt_workload_cfg['lstm_ratio']
    dlt_bert_ratio = dlt_workload_cfg['bert_ratio']
    dlt_others_ratio = dlt_workload_cfg['others_ratio']

    objective_cfg = para_cfg['objective']
    convergence_ratio = objective_cfg['convergence_ratio']
    accuracy_ratio = objective_cfg['accuracy_ratio']
    runtime_ratio = objective_cfg['runtime_ratio']

    random_seed = para_cfg['random_seed']

    #######################################################
    # generate the workload
    #######################################################

    wg = WorkloadGenerator(dlt_workload_size,
                           dlt_residual_ratio,
                           dlt_mobile_ratio,
                           dlt_lstm_ratio,
                           dlt_bert_ratio,
                           dlt_others_ratio,
                           convergence_ratio,
                           accuracy_ratio,
                           runtime_ratio,
                           random_seed)

    ml_workload = wg.generate_workload()

    if sched_name == 'laf':
        sched = LAF(path_file, para_file, knowledgebase_folder, ml_workload)
    elif sched_name == 'srf':
        sched = SRF(path_file, para_file, knowledgebase_folder, ml_workload)
    elif sched_name == 'bcf':
        sched = BCF(path_file, para_file, knowledgebase_folder, ml_workload)
    else:
        sched = Rotary(path_file, para_file, knowledgebase_folder, ml_workload)

    sched.run()


if __name__ == "__main__":
    main()
