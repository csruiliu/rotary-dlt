#!/bin/bash

MODEL_LIST="resnet"
BATCHSIZE_LIST="32 50 64 100"
TRAINSET_LIST="cifar10"
EPOCH_LIST="1 5 10 20"
OPT_LIST="Momentum"
LAYER_NUM_LIST="18 34 50 101 152"
LEARN_RATE_LIST="0.1 0.01 0.001 0.0001 0.00001"
ACTIVATION_LIST="relu"
DEVICE='gpu:0'

#MODEL_LIST="mobilenet resnet densenet mlp scn"
#BATCHSIZE_LIST="32 50 64 100"
#TRAINSET_LIST="imagenet cifar10"
#EPOCH_LIST="10 50 100"
#OPT_LIST="Adam SGD Adagrad Momentum"
#LAYER_NUM_LIST="18 34 50 101 152"
#LEARN_RATE_LIST="0.01 0.001 0.0001 0.00001"
#ACTIVATION_LIST="relu leaky_relu tanh sigmoid"
#DEVICE='/GPU:0'

REPEAT=2
FOLDER="exp-accuracy-results"
OUTPUT_FILE="accuracy-results.txt"

if [ -d "./${FOLDER}" ]
then
    echo "Directory ./${FOLDER} exists."
else
    mkdir ./${FOLDER}
fi

for midx in ${MODEL_LIST}
do
  for bidx in ${BATCHSIZE_LIST}
  do
    for tidx in ${TRAINSET_LIST}
    do
      for eidx in ${EPOCH_LIST}
      do
        for oidx in ${OPT_LIST}
        do
          for lidx in ${LAYER_NUM_LIST}
          do
            for ridx in ${LEARN_RATE_LIST}
            do
              for aidx in ${ACTIVATION_LIST}
              do
                CASE=${midx}_${bidx}_${tidx}_${eidx}_${oidx}_${lidx}_${ridx}_${aidx}
                echo ${CASE}
                for i in $(seq 1 ${REPEAT})
                do
                  echo "########################################" >> ./${FOLDER}/${CASE}-REPEAT${i}.txt
                  echo "================EXP ${i}================" >> ./${FOLDER}/${CASE}-REPEAT${i}.txt
                  echo "########################################" >> ./${FOLDER}/${CASE}-REPEAT${i}.txt
                  python3 accuracy_profiler.py -m ${midx} -b ${bidx} -t ${tidx} -e ${eidx} -o ${oidx} -l ${lidx} -r ${ridx} -a ${aidx} -d ${DEVICE} >> ./${FOLDER}/${CASE}-REPEAT${i}.txt
                done
                LEN=0
                SUM=0
                # echo "================EXP: ${CASE} ================" >> ./${FOLDER}/${OUTPUT_FILE}
                for j in $(seq 1 ${REPEAT})
                do
                  # echo "### REPEAT ${j} ###" >> ./${FOLDER}/${OUTPUT_FILE}
                  while read -r line
                  do
                    if [[ $line =~ ^"{\"model_name\": " ]]
                    then
                      BAK_LINE=${line}
                      PRE_ACC=${line#*"\"model_accuracy\": "}
                      ACC=${PRE_ACC%"}"*}
                      MODEL_ACC="$(echo "scale=9; ${ACC}" | bc)"
                      echo ${MODEL_ACC}
                      LEN=$((${LEN}+1))
                      SUM=$(echo "${SUM} + ${MODEL_ACC}" | bc)
                    fi
                  done < ./${FOLDER}/${CASE}-REPEAT${j}.txt
                done
                MODEL_NAME=${BAK_LINE%", \"model_accuracy\""*}
                echo ${MODEL_NAME}", \"model_accuracy\": ""$(echo "scale=4; ${SUM}/${LEN}" | bc | awk '{printf "%.4f", $0}')}," >> ./${FOLDER}/${OUTPUT_FILE}
              done
            done
          done
        done
      done
    done
  done
done

