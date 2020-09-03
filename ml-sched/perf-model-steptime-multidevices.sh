#!/bin/bash

CPU_MODEL_LIST="mobilenet"
CPU_MODEL_NUM_LIST="1"
CPU_MODEL_LAYER_NUM="1"
CPU_BATCH_SIZE="32"
CPU_OPT="Adam"
CPU_ACTIVATION="relu"
CPU_LEARN_RATE="0.001"

GPU_MODEL_LIST="resnet"
GPU_MODEL_NUM_LIST="1"
GPU_MODEL_LAYER_NUM="18"
GPU_BATCH_SIZE="32"
GPU_OPT="Adam"
GPU_ACTIVATION="relu"
GPU_LEARN_RATE="0.001"

TRAINSET_LIST="cifar10"
REPEAT=2
FOLDER="exp-results"

#CPU_MODEL_LIST="mobilenet resnet densenet mlp"
#CPU_MODEL_NUM="1 2 4 8 16"
#GPU_MODEL_LIST="mobilenet resnet densenet mlp"
#GPU_MODEL_NUM="1"
#BATCHSIZE_LIST="32 50 64 100 128"

TRAINSET_LIST="cifar10"
REPEAT=2
FOLDER="exp-results"
SUDOPWD=""

if [ -d "./${FOLDER}" ]
then
    echo "Directory ./${FOLDER} exists."
else
    mkdir ./${FOLDER}
fi

for tidx in ${TRAINSET_LIST}
do
  for cmidx in ${CPU_MODEL_LIST}
  do
    for clidx in ${CPU_MODEL_LAYER_NUM}
    do
      for cnidx in ${CPU_MODEL_NUM_LIST}
      do
        for cbidx in ${CPU_BATCH_SIZE}
        do
          for coidx in ${CPU_OPT}
          do
            for caidx in ${CPU_ACTIVATION}
            do
              for cridx in ${CPU_LEARN_RATE}
              do
                for gmidx in ${GPU_MODEL_LIST}
                do
                  for glidx in ${GPU_MODEL_LAYER_NUM}
                  do
                    for gnidx in ${GPU_MODEL_NUM_LIST}
                    do
                      for gbidx in ${GPU_BATCH_SIZE}
                      do
                        for goidx in ${GPU_OPT}
                        do
                          for gaidx in ${GPU_ACTIVATION}
                          do
                            for gridx in ${GPU_LEARN_RATE}
                            do
                              CASE=${cmidx}_${cnidx}_${cbidx}_${coidx}_${caidx}_${cridx}_${gmidx}_${gnidx}_${gbidx}_${goidx}_${gaidx}_${gridx}
                              #echo ${CASE}
                              #for i in $(seq 1 ${REPEAT})
                              #do
                              #  {
                              #    echo "########################################"
                              #    echo "================EXP ${i}================"
                              #    echo "########################################"
                              #  } >> ./${FOLDER}/${CASE}-REPEAT${i}.txt
                              #  python3 multidevices_time_profiler.py -cm ${cmidx} -cl ${clidx} -cn ${cnidx}\
                              #   -cb ${cbidx} -cr ${cridx} -ca ${caidx} -co ${coidx} -gm ${gmidx} -gl ${glidx}\
                              #   -gn ${gnidx} -gb ${gbidx} -gr ${gridx} -ga ${gaidx} -go ${goidx} -t ${tidx} >> ./${FOLDER}/${CASE}-REPEAT${i}.txt
                              #  rm -rf __pycache__
                              #  python3 clean_gpu_cache.py
                                #echo $SUDOPWD | sudo -S ./cleancache.sh
                              #done
                              LEN=0
                              SUM=0
                              PROGRESS_STEP_SUM=0
                              echo "================EXP: ${CASE} ================" >> ./${FOLDER}/all-results.txt
                              for j in $(seq 1 ${REPEAT})
                              do
                                echo "### REPEAT ${j} ###" >> ./${FOLDER}/all-results.txt
                                echo "total cpu job: ${cnidx}" >> ./${FOLDER}/all-results.txt
                                GPU_TIME=0
                                while read -r line
                                do
                                  if [[ $line =~ ^"GPU job average step time " ]]
                                  then
                                    BAK_LINE=$line
                                    AST=${line#*"]: "}
                                    GPU_TIME_PRE=${BAK_LINE#*[}
                                    GPU_TIME_POST=${GPU_TIME_PRE%]*}
                                    GPU_TIME="$(echo "scale=9; ${GPU_TIME_POST}" | bc)"
                                    echo ${GPU_TIME}
                                    echo ${AST}
                                    LEN=$((${LEN}+1))
                                    SUM=$(echo "${SUM} + ${AST}" | bc)
                                  fi
                                done < ./${FOLDER}/${CASE}-REPEAT${j}.txt

                                progress_dict=()

                                while read -r line
                                do
                                  if [[ $line =~ ^"**CPU JOB**: Proc-" ]]
                                  then
                                    BAK_LINE=${line}
                                    BAK_RES=${line}
                                    PROC_PRE=${line#*"**CPU JOB**: Proc-"}
                                    PROC=${PROC_PRE%","*}
                                    CPU_TIME_PRE=${BAK_RES#*[}
                                    CPU_TIME_POST=${CPU_TIME_PRE%]*}
                                    CPU_TIME="$(echo "scale=9; ${CPU_TIME_POST}" | bc)"
                                    if [ 1 -eq "$(echo "${CPU_TIME} > ${GPU_TIME}" | bc)" ] && [[ ${progress_dict[proc-${PROC}]} -ne 1 ]]
                                    then
                                      echo ${BAK_LINE} >> ./${FOLDER}/all-results.txt
                                      PROGRESS_STEP_PRE=${BAK_LINE#*"step "}
                                      PROGRESS_STEP=${PROGRESS_STEP_PRE%" / "*}
                                      PROGRESS_STEP_SUM=$((PROGRESS_STEP_SUM + PROGRESS_STEP - 1))
                                      progress_dict+=([proc-${PROC}]=1)
                                    fi
                                  fi
                                done < ./${FOLDER}/${CASE}-REPEAT${j}.txt
                              done
                              TOTAL_TRAIN_NUM=$((REPEAT * cnidx))
                              echo "CPU JOB AVG STEP PROGRESS=$(echo "scale=4; ${PROGRESS_STEP_SUM}/${TOTAL_TRAIN_NUM}" | bc)" >> ./${FOLDER}/all-results.txt
                              echo "### GPU JOB ###" >> ./${FOLDER}/all-results.txt
                              echo "GPU JOB AVG STEP=$(echo "scale=3; ${SUM}/${LEN}" | bc)" >> ./${FOLDER}/all-results.txt
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
        done
      done
    done
  done
done






