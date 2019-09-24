import numpy as np
from multiprocessing import Process, Pipe
from math import log, ceil, floor
from timeit import default_timer as timer
from hp_func import get_params, run_params, run_params_pack, evaluate_model, load_bin_raw

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

    def run_pack(self):
        for s in reversed(range(self.s_max + 1)):
            n = ceil(self.B / self.R / (s + 1) * (self.eta ** s))
            r = self.R * (self.eta ** (-s))
            T = self.get_hyperparams(n)

            for i in range(s + 1):
                n_i = floor(n * self.eta ** (-i))
                r_i = int(r * self.eta ** (i))
                print("\n*** {} bracket | {} configurations x {} iterations each ***".format(s, n_i, r_i))
                print("==============================================================")
                #val_acc = []
                
                #params_set = set()
                params_dict = dict()

                for t in T:
                    self.counter += 1
                    print("current run {}, current params: batch size {}, opt {} | \n".format(self.counter, t[0], t[1]))
                    #params_set.add(t[0])
                    if t[0] in params_dict:
                        params_dict[t[0]].append(t[1])
                    else:
                        params_dict[t[0]] = []
                        params_dict[t[0]].append(t[1])
                #print(params_set) 
                for bs, opt in params_dict.items():
                    parent_conn, child_conn = Pipe()
                    p = Process(target=self.run_hyperParams, args=(bs, opt, r_i, child_conn))
                    p.start()
                    acc = parent_conn.recv()
                    parent_conn.close()
                    p.join()
                    if len(acc) == 1:
                        result = {'acc':-1, 'counter':-1}
                        result['acc'] = acc
                        #esult['counter'] = self.counter
                        result['params'] = []
                        result['params'].append(bs)
                        result['params'].append(opt)

                        if self.best_acc < acc:
                            self.best_acc = acc
                            #self.best_counter = self.counter
                            print("best accuracy so far: {:.5f} \n".format(self.best_acc))
                        self.results.append(result)

                    else:
                        for idx, a in enumerate(acc):
                            result = {'acc':-1, 'counter':-1}
                            result['acc'] = a
                            #result['counter'] = self.counter
                            result['params'] = []
                            result['params'].append(bs)
                            result['params'].append(opt[idx])
                            
                            if self.best_acc < a:
                                self.best_acc = a
                                print("best accuracy so far: {:.5f} \n".format(self.best_acc))
                            self.results.append(result)
        return self.results        

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
    #evaluate_model()
    
    resource_conf = 27
    down_rate = 3
    hb = Hyperband(resource_conf, down_rate, get_params, run_params_pack)
    start_time = timer()
    results = hb.run_pack()
    end_time = timer()
    dur_time = end_time - start_time
    print("{} total, best:\n".format(len(results)))
    best_hp = sorted(results, key = lambda x: x['acc'])[-1]
    print(best_hp)
    print('total exp time:',dur_time)