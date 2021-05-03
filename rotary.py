import tensorflow as tf

import config.config_workload as cfg_workload
from workload.generator import WorkloadGenerator


def resource_sharing(workload):
    pass


def cost_model():
    pass


if __name__ == "__main__":
    n_gpu = 2

    wg = WorkloadGenerator(cfg_workload.workload_size,
                           cfg_workload.cv_light_ratio,
                           cfg_workload.cv_med_ratio,
                           cfg_workload.cv_heavy_ratio,
                           cfg_workload.nlp_light_ratio,
                           cfg_workload.nlp_med_ratio,
                           cfg_workload.nlp_heavy_ratio,
                           cfg_workload.convergence_ratio,
                           cfg_workload.accuracy_ratio,
                           cfg_workload.runtime_ratio,
                           cfg_workload.random_seed)

    ml_workload = wg.generate_workload()

    resource_sharing(ml_workload)