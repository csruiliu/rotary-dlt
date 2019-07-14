# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import csv
import re

import argparse
parser = argparse.ArgumentParser()
#parser.add_argument('-f', '--csvfile', type=str, help='identify a csv file to figure')
parser.add_argument("csvfile", help="identify a csv file to figure")
#parser.add_argument("outfile", help="identify the output filename")
args = parser.parse_args()

csvpath = args.csvfile
#outpath = args.outfile
outpath = csvpath.replace('.csv','.png')
expName = csvpath.split('.')[0]

mem_total = []
mem_free = []
mem_used = []
mem_util = []
gpu_util = []
count = 1
with open(csvpath,'r') as csvfile:
    next(csvfile)
    read = csv.reader(csvfile, delimiter=',')
    for row in read:
        mem_used.append(int(row[0].replace(' MiB','')))
        mem_free.append(int(row[1].replace(' MiB','')))
        mem_total.append(int(row[2].replace(' MiB','')))
        mem_util.append(int(row[3].replace(' %','')))
        gpu_util.append(int(row[4].replace(' %','')))
        count += 1

barWidth=1

x = np.arange(1, count)

#print(np.arange(0,16278,3255))

ax = plt.axes()
ax.tick_params(direction='in')
ax.grid(linestyle='--')
plt.title(expName)

fig = plt.gcf()
fig.set_size_inches(12, 4)

plt.subplot(1, 2, 1)

plt.bar(x, mem_used, width=barWidth, label='mem used')
plt.bar(x, mem_free, bottom=mem_used, width=barWidth, label='mem free')
plt.plot(x, mem_total, linewidth=3.0, color='g', label='mem total')

plt.yticks(np.arange(0,16278,3255), ('0%', '20%', '40%', '60%', '80%', '100%'))
plt.xlabel("Total Training Time (ms)")
plt.ylabel("GPU Memory")
plt.legend(loc='lower center')

plt.subplot(1, 2, 2)

plt.plot(x, gpu_util, color='b', label='gpu util')
plt.plot(x, mem_util, color='r', label='mem util')

plt.yticks(np.arange(0,103,20), ('0%', '20%', '40%', '60%', '80%', '100%'))
plt.xlabel("Total Training Time (ms)")
plt.ylabel("Utilization")
plt.legend(loc='lower center')

#plt.show()
plt.tight_layout()
fig.savefig(outpath,format='png')

