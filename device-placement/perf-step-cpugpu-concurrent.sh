#!/bin/bash
#CPU_MODEL_LIST="mobilenet resnet densenet mlp"
CPU_MODEL_LIST="mobilenet"
CPU_MODEL_NUM="2"
#CPU_MODEL_NUM="1 2 4 8 16"
GPU_MODEL_LIST="resnet"
GPU_MODEL_NUM="1"
BATCHSIZE_LIST="32"
#BATCHSIZE_LIST="32 50 64 100 128"
TRAINSET_LIST="imagenet"
REPEAT=5
FOLDER="exp-results"
SUDOPWD=""

for tidx in ${TRAINSET_LIST}
do
  for bidx in ${BATCHSIZE_LIST}
  do
    for gpum in ${GPU_MODEL_LIST}
    do
      for gpu_concur in ${GPU_MODEL_NUM}
      do
        for cpum in ${CPU_MODEL_LIST}
        do
          for cpu_concur in ${CPU_MODEL_NUM}
          do
            CASE=${tidx}_${bidx}_${gpum}_${gpu_concur}_${cpum}_${cpu_concur}
            echo ${CASE}
            for i in $(seq 1 ${REPEAT})
            do
              echo "########################################" >> ./${FOLDER}/${CASE}.txt
              echo "================EXP ${i}================" >> ./${FOLDER}/${CASE}.txt
              echo "########################################" >> ./${FOLDER}/${CASE}.txt
              python3 cpugpu_concurrent_profiler.py -cm ${cpum} -cn ${cpu_concur} -gm ${gpum} -gn ${gpu_concur} -b ${bidx} -d ${tidx}
              rm -rf __pycache__
              python3 clean_gpu_cache.py
              echo $SUDOPWD | sudo -S ./cleancache.sh
            done
          done
        done
      done
    done
  done
done




