# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import csv
import re

#csvpath = '/home/ruiliu/Development/mtml-tf/mt-exp/exp-result/exp-resnet.csv'
outpath = '/home/ruiliu/Development/mtml-tf/mt-exp/exp-result/gpu-batchsize-improve.pdf'

0.1666666667
0.173870334
0.179338843
0.1869863014

batch_size = [32,40,48,56,64,72,80]
bs_mlp = [39.8, 41.9, 43.5, 43.8, 44.3, 43.9, 45.2]
bs_mobile = [22.1, 23.3, 23.7, 24.3, 25.9, 26, 26.4]
batch_size_resnet = [32,40,48,56,64,72]
bs_resnet = [18.2, 18.3, 18.7, 18.9, 18.9, 18.9]
batch_size_densenet = [32,40,48,56]
bs_desenet = [16.7, 17.4, 17.9, 18.7]

#improve_resnet = [0, 21.1, 25, 27]
#improve_mobilenet = [0, 21.8, 30, 33.2]

#model_no_densenet = [1,2,3]
#improve_densenet = [0, 17.9, 21]

plt.figure(figsize=(6, 4), dpi=70)

plt.plot(batch_size, bs_mlp, color='royalblue', marker='^', linestyle='-', linewidth=2, markersize=8, label='MLP-3')
plt.plot(batch_size, bs_mobile, color='seagreen', marker='o', linestyle=':', linewidth=2, markersize=7, label='MobileNet')
plt.plot(batch_size_resnet, bs_resnet, color='orangered', marker='D', linestyle='-.', linewidth=2, markersize=6, label='ResNet-50')
plt.plot(batch_size_densenet, bs_desenet, color='goldenrod', marker='*', linestyle='--', linewidth=2, markersize=10, label='DenseNet-121')
plt.yticks(np.arange(0,103,20), ('0%', '20%', '40%', '60%', '80%', '100%'))
plt.xticks(batch_size)
plt.tick_params(axis='y',direction='in',labelsize=16) 
plt.tick_params(axis='x',direction='in',bottom='False',labelsize=16)

plt.xlabel("Batch Size", fontsize=16)
plt.ylabel("Improvement of Packing", fontsize=16)
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig(outpath,format='pdf',bbox_inches='tight', pad_inches=0.05)
#plt.show()
#print(x_pt)
#print(y_pt)



