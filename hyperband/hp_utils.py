import random
import numpy as np
import itertools
from operator import itemgetter 

def init_hp(n_conf):
	batch_size = np.arange(0,60,5)
	batch_size[0] = 1
	opt_conf = ['Adam','SGD']
	data_conf = ['Same','Diff']
	preprocess_list = ['Include','Not Include']
	all_conf = [batch_size,opt_conf,data_conf,preprocess_list]
	hp_conf = list(itertools.product(*all_conf))
	idx_list = np.random.choice(np.arange(0, len(hp_conf)), n_conf)
	rand_conf = itemgetter(*idx_list)(hp_conf)
	
	return rand_conf
	#print(rand_conf)
	#print(type(hp_conf)

def run_params(hyper_params, iterations):
	batch_size = hyper_params[0]
	opt = hyper_params[1]
	input_data = hyper_params[2]
	prep = hyper_params[3]
	
	print(hyper_params)
	print(iterations)