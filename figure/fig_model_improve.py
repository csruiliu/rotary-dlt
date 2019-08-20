# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import csv
import re




#csvpath = '/home/ruiliu/Development/mtml-tf/mt-exp/exp-result/exp-resnet.csv'
outpath = '/home/ruiliu/Development/mtml-tf/mt-exp/exp-result/exp-model-improve.png'


model_no = [1,2,3,4]
improve_mlp = [0, 39.7, 65, 74]
improve_resnet = [0, 21.1, 25, 27]
improve_mobilenet = [0, 21.8, 30, 33.2]

plt.figure(figsize=(6, 4), dpi=70)

plt.plot(model_no, improve_mlp, color='green', marker='^', linestyle='-', markersize=8, label='MLP')
plt.plot(model_no, improve_resnet, color='blue', marker='o', linestyle='-', markersize=8, label='ResNet')
plt.plot(model_no, improve_mobilenet, color='magenta', marker='*', linestyle='-', markersize=8, label='MobileNet')
plt.yticks(np.arange(0,103,20), ('0%', '20%', '40%', '60%', '80%', '100%'))
plt.xticks(np.arange(1,5,1))
plt.tick_params(axis='y',direction='in',labelsize=14) 
plt.tick_params(axis='x',direction='in',bottom='False',labelsize=14)

plt.xlabel("Number of Models", fontsize=11)
plt.ylabel("Improvement of Packing", fontsize=11)
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig(outpath,format='png')
#plt.show()
#print(x_pt)
#print(y_pt)



