#!/bin/bash

CPU_MODEL_LIST="mobilenet resnet densenet mlp"
CPU_MODEL_NUM="1 2 4 8 16"
GPU_MODEL_LIST="mobilenet resnet densenet mlp"
GPU_MODEL_NUM="1"
BATCHSIZE_LIST="32 50 64 100 128"
TRAINSET_LIST="imagenet cifar10"
REPEAT=5
FOLDER="exp-results"
SUDOPWD=""



