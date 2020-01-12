# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import csv
import re

#csvpath = '/home/ruiliu/Development/mtml-tf/mt-exp/exp-result/exp-resnet.csv'
outpath = '/home/ruiliu/Development/mtml-tf/figure/cpu-batchsize-improve.pdf'

batch_size = [32,40,48,56,64,72,80]
bs_mlp = [31.8, 32.7, 34.4, 34.8, 35.5, 36.1, 37.2]
bs_mobile = [40.8, 40.1, 40.1, 40, 39.2, 37, 36.6]
bs_resnet = [39.4, 40.3, 38, 38.5, 38.5, 39.7, 39]
bs_desenet = [40.3, 38.7, 37, 36.3, 36.9, 37, 38.8]

#improve_resnet = [0, 21.1, 25, 27]
#improve_mobilenet = [0, 21.8, 30, 33.2]

#model_no_densenet = [1,2,3]
#improve_densenet = [0, 17.9, 21]

plt.figure(figsize=(6, 4))

plt.plot(batch_size, bs_mlp, color='royalblue', marker='^', linestyle='-', linewidth=2, markersize=8, label='MLP-3')
plt.plot(batch_size, bs_mobile, color='seagreen', marker='o', linestyle=':', linewidth=2, markersize=7, label='MobileNet')
plt.plot(batch_size, bs_resnet, color='orangered', marker='D', linestyle='-.', linewidth=2, markersize=6, label='ResNet-50')
plt.plot(batch_size, bs_desenet, color='goldenrod', marker='*', linestyle='--', linewidth=2, markersize=10, label='DenseNet-121')
plt.yticks(np.arange(0,103,20), ('0%', '20%', '40%', '60%', '80%', '100%'))
plt.xticks(batch_size)
plt.tick_params(axis='y',direction='in',labelsize=20) 
plt.tick_params(axis='x',direction='in',bottom='False',labelsize=20)

plt.xlabel("Batch Size", fontsize=22)
plt.ylabel("Improvement of Packing", fontsize=21)
plt.legend(loc='upper left', fontsize=16)
plt.tight_layout()
plt.savefig(outpath,format='pdf',bbox_inches='tight', pad_inches=0.05)
#plt.show()
#print(x_pt)
#print(y_pt)



