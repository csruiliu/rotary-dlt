#!/bin/bash

DEVICE_LIST="gpu:0"
MODEL_TYPE_LIST="resnet"
MODEL_NUM_LIST="1"
MODEL_LAYER_NUM_LIST="18"
EPOCH_LIST="1"
BATCHSIZE_LIST="32"
LEARNRATE_LIST="0.0001"
OPT_LIST="Adam"
ACTIVATION_LIST="relu"

TRAINSET_LIST="cifar10"

REPEAT=2
FOLDER="exp-steptime-singledevice-results"
OUTPUT_FILE="all-results.txt"

if [ -d "./${FOLDER}" ]
then
    echo "Directory ./${FOLDER} exists."
else
    mkdir ./${FOLDER}
fi

for didx in ${DEVICE_LIST}
do
  for mtidx in ${MODEL_TYPE_LIST}
  do
    for mnidx in ${MODEL_NUM_LIST}
    do
      for mlidx in ${MODEL_LAYER_NUM_LIST}
      do
        for eidx in ${EPOCH_LIST}
        do
          for bidx in ${BATCHSIZE_LIST}
          do
            for ridx in ${LEARNRATE_LIST}
            do
              for oidx in ${OPT_LIST}
              do
                for aidx in ${ACTIVATION_LIST}
                do
                  for tidx in ${TRAINSET_LIST}
                  do
                    CASE=${mtidx}_${mnidx}_${mlidx}_${bidx}_${ridx}_${oidx}_${aidx}_${tidx}_${didx}
                    echo ${CASE}
                    for i in $(seq 1 ${REPEAT})
                    do
                      rm -rf __pycache__
                      python3 clean_gpu_cache.py
                      python3 profiler_steptime_singledevice.py -m ${mtidx} -n ${mnidx} -e ${eidx} -b ${bidx} -r ${ridx} -o ${oidx} -a ${aidx} -l ${mlidx} -t ${tidx} -d ${didx} >> ./${FOLDER}/${CASE}-REPEAT${i}.txt
                    done
                    SUM=0
                    #echo "================EXP: ${CASE} ================" >> ./${FOLDER}/all-results.txt
                    for j in $(seq 1 ${REPEAT})
                    do
                      #echo "### REPEAT ${j} ###" >> ./${FOLDER}/all-results.txt
                      #echo "total job: ${mnidx} on ${didx}" >> ./${FOLDER}/all-results.txt
                      while read -r line
                      do
                        if [[ $line =~ ^"{\"data_cost\": " ]]
                        then
                          BAK_LINE=$line
                          TIME_PRE=${line#*"model_step_time\":"}
                          TIME_POST=${TIME_PRE%"}"*}
                          STEP_TIME="$(echo "scale=9; ${TIME_POST}" | bc)"
                          #echo "STEP_TIME: ${STEP_TIME}">> ./${FOLDER}/all-results.txt
                          SUM=$(echo "${SUM} + ${STEP_TIME}" | bc)
                        fi
                      done < ./${FOLDER}/${CASE}-REPEAT${j}.txt
                    done
                    #echo "#######################" >> ./${FOLDER}/all-results.txt
                    MODEL_NAME=${BAK_LINE%", \"model_step_time\""*}
                    echo ${MODEL_NAME}", \"model_step_time\": ""$(echo "scale=4; ${SUM}/${REPEAT}" | bc | awk '{printf "%.4f", $0}')}," >> ./${FOLDER}/${OUTPUT_FILE}
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done










