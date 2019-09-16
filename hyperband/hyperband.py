import numpy as np
from math import log, ceil, floor
from timeit import default_timer as timer
import hp_utils

class Hyperband:
	def __init__(self, getHyperPara, runHyperPara):
		# maximun budget for single configuration, i.e., maximum iterations per configuration in example
		self.R = 81
		# defines configuration downsampling rate (default = 3)
		self.eta = 3		
		# control how many runs
		self.s_max = floor(log(self.R, self.eta))
		# maximun budget for all configurations
		self.B = (self.s_max + 1) * self.R
		# list of dicts
		self.results = []	
		self.counter = 0
		self.best_loss = np.inf
		self.best_counter = -1
		self.get_hyperparams = getHyperPara
		self.run_hyperParams = runHyperPara

	def run(self, skip_last=0, dry_run=True):
		for s in reversed(range(self.s_max + 1)):
			n = ceil(self.B / self.R / (s + 1) * (self.eta ** s))
			r = self.R * self.eta ** (-s)
			T = self.get_hyperparams(n)
			
			for i in range(s + 1):
				n_i = floor(n * self.eta ** (-i))
				r_i = int(r * self.eta ** (i))
				print("\n*** {} bracket | {} configurations x {} iterations each ***".format(s, n_i, r_i))
				print("==============================================================")

				val_losses = []

				for t in T:
					self.counter += 1
					
					start_time = timer()

					if dry_run:
						self.run_hyperParams(t, r_i)
						result = {'loss':np.random.random(),'auc':np.random.random()}
					else:
						result = self.run_hyperParams(n_iterations, t)	
					
					# check the results format and if loss is generated 
					assert(type(result) == dict)
					assert('loss' in result)
					
					end_time = timer()

					seconds = end_time - start_time
					#print("\n{} seconds.".format(seconds))

					loss = result['loss']	
					val_losses.append(loss)
					
					if loss < self.best_loss:
						self.best_loss = loss
						self.best_counter = self.counter

					print("current run {}, loss: {:.5f} | lowest loss so far: {:.5f} (run {})\n".format(self.counter, loss, self.best_loss, self.best_counter))

					result['counter'] = self.counter
					result['seconds'] = seconds
					result['params'] = t
					# r_i is the resource, i.e., epochs to run the current configuration t in the example
					result['resources'] = r_i

					self.results.append(result)

				indices = np.argsort(val_losses)
				
				T = [T[i] for i in indices]
				T = T[0:floor(n_i / self.eta)]
				
				#print(indices)

				#T = T[0:int(n_i / self.eta)]
		
		#print("best result=", self.best_loss)
		#print("best counter=", self.best_counter)
		return self.results

if __name__ == "__main__":
	#hp_utils.init_hp(10)
	hb = Hyperband(hp_utils.init_hp, hp_utils.run_params)
	results = hb.run()
	print("{} total, best:\n".format(len(results)))
	best_hp = sorted(results, key = lambda x: x['loss'])[0]
	print(best_hp)