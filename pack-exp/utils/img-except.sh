#!/bin/bash
input="/home/ruiliu/Development/mtml-tf/dataset/imagenet10k"
for entry in "$input"/*
do
  echo $entry
  python3 img_check.py -i $entry 
done