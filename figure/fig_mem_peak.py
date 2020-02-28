# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

outpath = '/home/ruiliu/Development/mtml-tf/figure/gpu-mem.pdf'

b32 = (3010, 1893, 3501, 4229, 3808, 3762)
b48 = (3010, 2809, 5062, 6307, 5212, 5503)
b64 = (3010, 3725, 6624, 8386, 6823, 7286)
b80 = (3010, 4578, 8185, 10464, 8434, 9093)

ind = np.arange(0,18,3) 
width = 0.4

plt.figure(figsize=(15, 6))
    
plt.bar(ind, b32, width, label='Batch Size 32', edgecolor='k', color='goldenrod', hatch='//')
plt.bar(ind + width, b48, width, label='Batch Size 48', edgecolor='k', color='forestgreen', hatch='.')
plt.bar(ind + 2*width, b64, width, label='Batch Size 56', edgecolor='k', color='royalblue', hatch='x')
plt.bar(ind + 3*width, b80, width, label='Batch Size 80', edgecolor='k', color='tomato', hatch='O')

plt.tick_params(axis='y',direction='in',labelsize=24) 
plt.tick_params(axis='x',direction='in',bottom='False',labelsize=22)

#plt.xlabel(r'$T_s(Seq)$', fontsize=16)
plt.ylabel('Peak Memory Usage (MB)', fontsize=26)

plt.xticks(ind + 1.5*width, ('MLP-3', 'MobileNet', 'ResNet-50', 'DenseNet-121', 'Pack \n (MLP+ResNet)', 'Pack \n (2xMobileNet)'))
plt.legend(loc='best',fontsize=24)
plt.tight_layout()
plt.savefig(outpath, format='pdf', bbox_inches='tight', pad_inches=0.05)

#plt.xticks(np.arange(0,outputSize+1,outputSize/4))
#plt.yticks(np.arange(0,outputSize+1,outputSize/4))


#plt.grid(linestyle='--')
#plt.title(title + ' (' + modelTitle + ')', fontsize=16, pad=10)




