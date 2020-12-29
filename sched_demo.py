import argparse
from relish.evaluation.roundrobin import roundrobin_run
from relish.evaluation.tetrisched_greedy import tetrisched_run
from relish.evaluation.hypersched import hypersched_run
from relish.evaluation.relish_evaluation import relish_run


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--evaluation', action='store', type=str,
                        choices=['roundrobin', 'hyperband', 'tetrisched', 'relish'],
                        help='the training pattern')
    args = parser.parse_args()
    sched_eval = args.evaluation

    if sched_eval == 'roundrobin':
        roundrobin_run()
    elif sched_eval == 'hyperband':
        hypersched_run()
    elif sched_eval == 'tetrisched':
        tetrisched_run()
    elif sched_eval == 'relish':
        relish_run()
    else:
        raise ValueError('the schedule evaluation is not recognized')