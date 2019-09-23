import numpy as np
from multiprocessing import Process, Pipe
from math import log, ceil, floor
from timeit import default_timer as timer
from hp_func import get_params, run_params, evaluate_model, load_bin_raw

class Hyperband:

    def __init__(self, resourceConf, downRate, getHyperPara, runHyperPara):
        # maximun budget for single configuration, i.e., maximum iterations per configuration in example
        self.R = resourceConf
        # defines configuration downsampling rate (default = 3)
        self.eta = downRate
        # control how many runs
        self.s_max = floor(log(self.R, self.eta))
        # maximun budget for all configurations
        self.B = (self.s_max + 1) * self.R
        # list of results
        self.results = []
        
        self.counter = 0
        self.best_acc = np.NINF
        self.best_counter = -1

        self.get_hyperparams = getHyperPara
        self.run_hyperParams = runHyperPara

    def run_fake(self):
        for s in reversed(range(self.s_max + 1)):
            n = ceil(self.B / self.R / (s + 1) * (self.eta ** s))
            r = self.R * (self.eta ** (-s))
            T = self.get_hyperparams(n)

            for i in range(s + 1):
                n_i = floor(n * self.eta ** (-i))
                r_i = int(r * self.eta ** (i))
                print("\n*** {} bracket | {} configurations x {} iterations each ***".format(s, n_i, r_i))
                print("==============================================================")

                # record all accuracy of current run for sorting
                val_acc = []

                for t in T:
                    result = {'acc':-1, 'counter':-1}
                    self.counter += 1
                    # generate random accuracy
                    acc = np.random.random()
                    val_acc.append(acc)

                    result['acc'] = acc
                    result['counter'] = self.counter
                    result['params'] = t

                    # record the best result
                    if self.best_acc < acc:
                        self.best_acc = acc
                        self.best_counter = self.counter
                    
                    print("current run {}, accuracy: {:.5f} | best accuracy so far: {:.5f} (run {})\n".format(self.counter, acc, self.best_acc, self.best_counter))

                    self.results.append(result)
                    
                # sort the result
                indices = np.argsort(val_acc)
                
                T = [T[i] for i in indices]
                T = T[0:floor(n_i / self.eta)]
        
        return self.results

    def run(self):
        for s in reversed(range(self.s_max + 1)):
            n = ceil(self.B / self.R / (s + 1) * (self.eta ** s))
            r = self.R * self.eta ** (-s)
            T = self.get_hyperparams(n)

            for i in range(s + 1):
                n_i = floor(n * self.eta ** (-i))
                r_i = int(r * self.eta ** (i))
                print("\n*** {} bracket | {} configurations x {} iterations each ***".format(s, n_i, r_i))
                print("==============================================================")

                # init list of accuracy of confs                
                list_acc = []

                for t in T:
                    result = {'acc':-1, 'counter':-1}
                    self.counter += 1

                    # use process to run multipe models
                    parent_conn, child_conn = Pipe()
                    p = Process(target=self.run_hyperParams, args=(t, r_i, child_conn))
                    p.start()
                    acc = parent_conn.recv()
                    result['acc'] = acc
                    parent_conn.close()
                    p.join()

                    list_acc.append(acc)

                    if self.best_acc < acc:
                        self.best_acc = acc
                        self.best_counter = self.counter

                    result['counter'] = self.counter
                    result['params'] = t

                    print("current run {}, accuracy: {:.5f} | best accuracy so far: {:.5f} (run {})\n".format(self.counter, acc, self.best_acc, self.best_counter))

                    self.results.append(result)
                
                indices = np.argsort(list_acc)

                T = [T[i] for i in indices]
                T = T[0:floor(n_i / self.eta)]

            return self.results

if __name__ == "__main__":
    load_bin_raw('/home/ruiliu/Development/mtml-tf/dataset/imagenet1k-aaa.bin')
    #evaluate_model()
    #resource_conf = 27
    #down_rate = 3
    #hb = Hyperband(resource_conf, down_rate, get_params, run_params)
    #results = hb.run_fake()
    
    #print("{} total, best:\n".format(len(results)))
    #best_hp = sorted(results, key = lambda x: x['acc'])[-1]
    #print(best_hp)
