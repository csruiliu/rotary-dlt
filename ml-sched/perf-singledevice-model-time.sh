#!/bin/bash

DEVICE_LIST="cpu"
MODEL_LIST="mobilenet resnet densenet mlp scn"
LAYER_NUM="1"
MODEL_NUM="1 2 4 8 16"
BATCHSIZE_LIST="32"
LEARNRATE_LIST="0.0001"
OPT_LIST="Adam"
ACTIVATION_LIST="relu"
TRAINSET_LIST="cifar10 imagenet"

REPEAT=2
FOLDER="exp-results"

if [ -d "./${FOLDER}" ]
then
    echo "Directory ./${FOLDER} exists."
else
    mkdir ./${FOLDER}
fi

for didx in ${DEVICE_LIST}
do
  for midx in ${MODEL_LIST}
  do
    for nidx in ${MODEL_NUM}
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
                CASE=${didx}_${midx}_${nidx}_${bidx}_${ridx}_${oidx}_${aidx}_${tidx}
                for i in $(seq 1 ${REPEAT})
                do
                  rm -rf __pycache__
                  python3 clean_gpu_cache.py
                  if [[ ${midx} == "scn" ]] || [[ ${midx} == "mlp" ]]
                  then
                    for lidx in ${LAYER_NUM}
                    do
                      python3 singledevice_time_profiler.py -d ${didx} -m ${midx} -l ${lidx} -n ${nidx} -b ${bidx} -r ${ridx} -o ${oidx} -a ${aidx} -t ${tidx} >> ./${FOLDER}/${CASE}-REPEAT${i}.txt
                    done
                  else
                    python3 singledevice_time_profiler.py -d ${didx} -m ${midx} -n ${nidx} -b ${bidx} -r ${ridx} -o ${oidx} -a ${aidx} -t ${tidx} >> ./${FOLDER}/${CASE}-REPEAT${i}.txt
                  fi
                done
                LEN=0
                SUM=0
                echo "================EXP: ${CASE} ================" >> ./${FOLDER}/all-results.txt
                for j in $(seq 1 ${REPEAT})
                do
                  echo "### REPEAT ${j} ###" >> ./${FOLDER}/all-results.txt
                  echo "total job: ${nidx} on ${didx}" >> ./${FOLDER}/all-results.txt
                  while read -r line
                  do
                    if [[ $line =~ ^"job average step time" ]]
                    then
                      TIME_PRE=${line#*"["}
                      TIME_POST=${TIME_PRE%"]"*}
                      STEP_TIME="$(echo "scale=9; ${TIME_POST}" | bc)"
                      echo "STEP_TIME: ${STEP_TIME}">> ./${FOLDER}/all-results.txt
                      LEN=$((${LEN}+1))
                      SUM=$(echo "${SUM} + ${STEP_TIME}" | bc)
                    fi
                  done < ./${FOLDER}/${CASE}-REPEAT${j}.txt
                done
                echo "#######################" >> ./${FOLDER}/all-results.txt
                echo "${nidx} ${midx} job on ${didx} avg step=$(echo "scale=6; ${SUM}/${LEN}" | bc)" >> ./${FOLDER}/all-results.txt
                echo "#######################" >> ./${FOLDER}/all-results.txt
              done
            done
          done
        done
      done
    done
  done
done





