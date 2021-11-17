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
    SCHED_OTHERS = 'others'
