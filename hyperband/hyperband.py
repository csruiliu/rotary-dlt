import numpy as np
from multiprocessing import Process, Pipe
#from multiprocessing import Pool
from math import log, ceil, floor
from timeit import default_timer as timer
import hp_utils

class Hyperband:

    def __init__(self, getHyperPara, runHyperPara):
        # maximun budget for single configuration, i.e., maximum iterations per configuration in example
        self.R = 27
        # defines configuration downsampling rate (default = 3)
        self.eta = 3
        # control how many runs
        self.s_max = floor(log(self.R, self.eta))
        # maximun budget for all configurations
        self.B = (self.s_max + 1) * self.R
        # list of dicts
        self.results = []
        self.counter = 0
        self.best_acc = np.NINF
        self.best_counter = -1
        self.get_hyperparams = getHyperPara
        self.run_hyperParams = runHyperPara
        
    def run(self, skip_last=0, dry_run=False):
        
        for s in reversed(range(self.s_max + 1)):
            n = ceil(self.B / self.R / (s + 1) * (self.eta ** s))
            r = self.R * self.eta ** (-s)
            T = self.get_hyperparams(n)

            for i in range(s + 1):
                n_i = floor(n * self.eta ** (-i))
                r_i = int(r * self.eta ** (i))
                print("\n*** {} bracket | {} configurations x {} iterations each ***".format(s, n_i, r_i))
                print("==============================================================")
                
                val_acc = []

                for t in T:
                    result = {'acc':-1, 'counter':-1}
                    self.counter += 1                    
                    if dry_run:
                        acc = np.random.random()
                        result['acc'] = acc
                    else:
                        parent_conn, child_conn = Pipe()
                        p = Process(target=self.run_hyperParams, args=(t, r_i, child_conn))
                        p.start()
                        acc = parent_conn.recv()
                        result['acc'] = acc
                        parent_conn.close()
                        p.join()

                        #print(data)
                        #child_conn.close()
                        #result = {'acc':np.random.random()}
                    
                    # check the results format and if loss is generated
                    #assert(type(result) == dict)
                    #assert('acc' in result)

                    val_acc.append(acc)
                    
                    if self.best_acc < acc:
                        self.best_acc = acc
                        self.best_counter = self.counter
                    
                    result['counter'] = self.counter
                    result['params'] = t
                    
                    print("current run {}, accuracy: {:.5f} | best accuracy so far: {:.5f} (run {})\n".format(self.counter, acc, self.best_acc, self.best_counter))

                    # r_i is the resource, i.e., epochs to run the current configuration t in the example
                    # result['resources'] = r_i
                    self.results.append(result)
                    indices = np.argsort(val_acc)

                T = [T[i] for i in indices]
                T = T[0:floor(n_i / self.eta)]

        return self.results

if __name__ == "__main__":
	#hp_utils.init_hp(10)
	hb = Hyperband(hp_utils.init_hp, hp_utils.run_params)
	results = hb.run()
	print("{} total, best:\n".format(len(results)))
	best_hp = sorted(results, key = lambda x: x['acc'])[-1]
	print(best_hp)
