# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import csv
import re

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--samemodel', action='store_true', default=False, help='pack same model to train or not')
parser.add_argument('-p', '--preproc', action='store_true', default=False, help='use preproc to transform the data before training or not')
parser.add_argument('-d', '--samedata', action='store_true', default=False, help='use same training batch data or not')
parser.add_argument('-o', '--sameoptimizer', action='store_true', default=False, help='use same optimizer or not')
parser.add_argument('-b', '--samebatchsize', action='store_true', default=False, help='use same batch size or not')
parser.add_argument('-s', '--size', type=int, default=1200, help='identify the size of img')
args = parser.parse_args()

sameBatchSize = args.samebatchsize
sameModel = args.samemodel
preproc = args.preproc
sameTrainData = args.samedata
sameOptimizer = args.sameoptimizer
outputSize = args.size

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


csvpath = '/home/ruiliu/Development/mtml-tf/mt-exp/exp-result/exp-resnet.csv'
outpath = '/home/ruiliu/Development/mtml-tf/mt-exp/exp-result/exp-seq-pack-'+path+'-resnet.png'

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


plt.figure(figsize=(6, 4), dpi=70)

plt.plot(positive_list_x, positive_list_y, color='green', marker='^', linestyle='', markersize=10)

plt.plot(negative_list_x, negative_list_y, color='red', marker='v', linestyle='', markersize=10)

plt.axis([0, outputSize, 0, outputSize])
a = np.linspace(0, outputSize, 1000)
plt.plot(a, a, '-b')
plt.tick_params(axis='y',direction='in',labelsize=14) 
plt.tick_params(axis='x',direction='in',bottom='False',labelsize=14)
plt.grid(linestyle='--')
plt.title(title + ' (ResNet)')
plt.xlabel("Time of one step of training two models in sequence (w/o switch overhead)", fontsize=11)
plt.ylabel("Time of one step of packing two models", fontsize=11)
plt.tight_layout()
plt.savefig(outpath,format='png')

