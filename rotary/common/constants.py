class JobStatus:
    INCOMPLETE = 0
    COMPLETE_UNATTAIN = 1
    COMPLETE_ATTAIN = 2


class JobSLO:
    ACCURACY = 'accuracy'
    RUNTIME = 'runtime'
    CONVERGENCE = 'convergence'


class SchedType:
    SCHED_ACCURACY = 'acc'
    SCHED_CONVERGENCE = 'converge'
    SCHED_RUNTIME = 'runtime'
    SCHED_OTHERS = 'others'
    SCHED_TRIAL = 'trial'
    SCHED_ROTARY = 'rotary'
