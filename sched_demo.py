
import tools.reward_func as reward_function


'''
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
'''