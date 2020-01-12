# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import csv
import re

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--title', type=str, default='MobileNet', help='model')
parser.add_argument('-m', '--samemodel', action='store_true', default=False, help='pack same model to train or not')
parser.add_argument('-p', '--preproc', action='store_true', default=False, help='use preproc to transform the data before training or not')
parser.add_argument('-d', '--samedata', action='store_true', default=False, help='use same training batch data or not')
parser.add_argument('-o', '--sameoptimizer', action='store_true', default=False, help='use same optimizer or not')
parser.add_argument('-b', '--samebatchsize', action='store_true', default=False, help='use same batch size or not')
parser.add_argument('-c', '--usecpu', action='store_true', default=False, help='use cpu or not')
parser.add_argument('-s', '--size', type=int, default=800, help='identify the size of img')
args = parser.parse_args()

modelTitle = args.title
sameBatchSize = args.samebatchsize
sameModel = args.samemodel
preproc = args.preproc
sameTrainData = args.samedata
sameOptimizer = args.sameoptimizer
outputSize = args.size
useCPU = args.usecpu

if sameModel:
    path = 'model'
    ridx = 0
    title = 'Same Model'
    mark = 'same'
elif sameTrainData:
    path = 'data'
    ridx = 1
    title = 'Same Training Data'
    mark = 'same'
elif preproc:
    path = 'preproc'
    ridx = 2
    title = 'Include Preprocessing'
    mark = 'yes'
elif sameOptimizer:
    path = 'optimizer'
    ridx = 3
    title = 'Same Optimizer'
    mark = 'same'
elif sameBatchSize:
    path = 'batchsize'
    ridx = 4
    title = 'Same Batch Size'
    mark = 'same'

if useCPU:
    csvpath = '/home/ruiliu/Development/mtml-tf/mt-exp/exp-result/exp-'+modelTitle+'-cpu.csv'
    outpath = '/home/ruiliu/Development/mtml-tf/mt-exp/exp-result/cpu-seq-pack-' + modelTitle + '-' + path + '.pdf'
else:
    csvpath = '/home/ruiliu/Development/mtml-tf/mt-exp/exp-result/exp-'+modelTitle+'.csv'
    outpath = '/home/ruiliu/Development/mtml-tf/mt-exp/exp-result/gpu-seq-pack-' + modelTitle + '-' + path + '.pdf'


positive_list_x = []
positive_list_y = []

negative_list_x = []
negative_list_y = []

with open(csvpath,'r') as csvfile:
    read = csv.reader(csvfile, delimiter=',')
    for row in read:
        if row[ridx] == mark:
            seq_x = int(row[6])
            pack_y = int(row[5])
            if pack_y > seq_x:
                negative_list_x.append(seq_x)
                negative_list_y.append(pack_y)
            else:
                positive_list_x.append(seq_x)
                positive_list_y.append(pack_y)
#print(x_pt)
#print(y_pt)


plt.figure(figsize=(5, 3.5), dpi=70)

plt.plot(positive_list_x, positive_list_y, color='green', marker='^', linestyle='', markersize=10)

plt.plot(negative_list_x, negative_list_y, color='red', marker='v', linestyle='', markersize=10)

plt.axis([0, outputSize, 0, outputSize])

a = np.linspace(0, outputSize, 1000)

plt.plot(a, a, '-b')

if useCPU:
    plt.plot(a, 0.8*a, color='royalblue', linestyle='-.', linewidth=0.9)
    plt.plot(a, 1.2*a, color='royalblue', linestyle='-.', linewidth=0.9)

    plt.annotate(r'$-20\%$', color='royalblue', xy=(0.76*outputSize, 0.88*outputSize),fontsize=8)
    plt.annotate(r'$+20\%$', color='royalblue', xy=(0.86*outputSize, 0.65*outputSize),fontsize=8)

    plt.plot(a, 0.6*a, color='royalblue', linestyle=':')
    plt.plot(a, 1.4*a, color='royalblue', linestyle=':')
    plt.annotate(r'$-40\%$', color='royalblue', xy=(0.67*outputSize, 0.92*outputSize),fontsize=8)
    plt.annotate(r'$+40\%$', color='royalblue', xy=(0.8*outputSize, 0.42*outputSize),fontsize=8)
else:
    plt.plot(a, 0.8*a, color='royalblue', linestyle='-.',linewidth=0.9)
    plt.plot(a, 1.2*a, color='royalblue', linestyle='-.',linewidth=0.9)

    plt.annotate(r'$-20\%$', color='royalblue', xy=(0.6*outputSize, 0.88*outputSize))
    plt.annotate(r'$+20\%$', color='royalblue', xy=(0.86*outputSize, 0.65*outputSize))
    
plt.xticks(np.arange(0,outputSize+1,outputSize/4))
plt.yticks(np.arange(0,outputSize+1,outputSize/4))

plt.tick_params(axis='y',direction='in',labelsize=16) 
plt.tick_params(axis='x',direction='in',bottom='False',labelsize=16)
plt.grid(linestyle='--')
plt.title(title + ' (' + modelTitle + ')', fontsize=16, pad=10)
plt.xlabel(r'$T_s(Seq)$', fontsize=16)
plt.ylabel(r'$T_s(Pack)$', fontsize=16)
plt.tight_layout()
plt.savefig(outpath,format='pdf', bbox_inches='tight', pad_inches=0.05)

