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
REPEAT=2
FOLDER="exp-results"
SUDOPWD=""

if [ -d "./exp-results" ]
then
    echo "Directory ./exp-results exists."
else
    mkdir ./exp-results
fi

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
              echo "########################################" >> ./${FOLDER}/${CASE}-REPEAT${i}.txt
              echo "================EXP ${i}================" >> ./${FOLDER}/${CASE}-REPEAT${i}.txt
              echo "########################################" >> ./${FOLDER}/${CASE}-REPEAT${i}.txt
              python3 cpugpu_concurrent_profiler.py -cm ${cpum} -cn ${cpu_concur} -gm ${gpum} -gn ${gpu_concur} -b ${bidx} -d ${tidx} >> ./${FOLDER}/${CASE}-REPEAT${i}.txt
              rm -rf __pycache__
              python3 clean_gpu_cache.py
              echo $SUDOPWD | sudo -S ./cleancache.sh
            done
            LEN=0
            SUM=0
            echo "================EXP: ${CASE} ================" >> ./${FOLDER}/all-results.txt
            for j in $(seq 1 ${REPEAT})
            do
              PROG_FLAG=0
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
              while read -r line
              do
                if [[ $line =~ ^"**CPU JOB**" ]] && [ ${PROG_FLAG} -eq 0 ]
                then
                  BAK_LINE=${line}
                  CPU_TIME_PRE=${line#*[}
                  CPU_TIME_POST=${CPU_TIME_PRE%]*}
                  CPU_TIME="$(echo "scale=9; ${CPU_TIME_POST}" | bc)"
                  if [ 1 -eq "$(echo "${CPU_TIME} > ${GPU_TIME}" | bc)" ]
                  then
                    echo ${BAK_LINE} >> ./${FOLDER}/all-results.txt
                    PROG_FLAG=1
                  fi
                elif [[ $line =~ ^"cpu job starts..." ]]
                then
                  PROG_FLAG=0
                fi
              done < ./${FOLDER}/${CASE}-REPEAT${j}.txt
            done
            echo AVG="$(echo "scale=3; ${SUM}/${LEN}" | bc)" >> ./${FOLDER}/all-results.txt
          done
        done
      done
    done
  done
done




