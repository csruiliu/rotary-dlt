#!/bin/sh

MODEL_LIST="alexnet efficientnet inception lenet mobilenet mobilenetv2 squeezenet xception zfnet"
BATCH_SIZE_LIST="32 64 128 256"
OPT_LIST="SGD Adam Adagrad Momentum"
LEARN_RATE_LIST="0.1 0.01 0.001 0.0001 0.00001"
EPOCH=20
PROFILE="accuracy"

for model in ${MODEL_LIST}
do
  for batch in ${BATCH_SIZE_LIST}
  do
    for opt in ${OPT_LIST}
    do
      for lr in ${LEARN_RATE_LIST}
      do
        rm -rf __pycache__
        python3 ml_profiler.py -p ${PROFILE} -m ${model} -b ${batch} -o ${opt} -r ${lr} -e ${EPOCH}
      done
    done
  done
done

MODEL_LIST="resnet densenet resnext vgg shufflenet shufflenetv2"
BATCH_SIZE_LIST="32 64 128 256"
OPT_LIST="SGD Adam Adagrad Momentum"
LEARN_RATE_LIST="0.1 0.01 0.001 0.0001 0.00001"
EPOCH=20
PROFILE="accuracy"

for batch in ${BATCH_SIZE_LIST}
do
  for opt in ${OPT_LIST}
  do
    for lr in ${LEARN_RATE_LIST}
    do
      for model in ${MODEL_LIST}
      do
        rm -rf __pycache__
        if [ "${model}" = "resnet" ]; then
          LAYER_LIST="18 34 50 101 152"
          for layer in ${LAYER_LIST}
          do
            python3 ml_profiler.py -p ${PROFILE} -m ${model} -l ${layer} -b ${batch} -o ${opt} -r ${lr} -e ${EPOCH}
          done
        elif [ "${model}" = "densenet" ]; then
          LAYER_LIST="121 169 201 264"
          for layer in ${LAYER_LIST}
          do
            python3 ml_profiler.py -p ${PROFILE} -m ${model} -l ${layer} -b ${batch} -o ${opt} -r ${lr} -e ${EPOCH}
          done
        elif [ "${model}" = "vgg" ]; then
          LAYER_LIST="11 13 16 19"
          for layer in ${LAYER_LIST}
          do
            python3 ml_profiler.py -p ${PROFILE} -m ${model} -l ${layer} -b ${batch} -o ${opt} -r ${lr} -e ${EPOCH}
          done
        elif [ "${model}" = "shufflenet" ]; then
          GROUP_LIST="2 3 4 8"
          for group in ${GROUP_LIST}
          do
            python3 ml_profiler.py -p ${PROFILE} -m ${model} -g ${group} -b ${batch} -o ${opt} -r ${lr} -e ${EPOCH}
          done
        elif [ "${model}" = "shufflenetv2" ]; then
          CPLX_LIST="0.5 1 1.5 2"
          for complex in ${CPLX_LIST}
          do
            python3 ml_profiler.py -p ${PROFILE} -m ${model} -x ${complex} -b ${batch} -o ${opt} -r ${lr} -e ${EPOCH}
          done
        elif [ "${model}" = "resnext" ]; then
          CARD_LIST="1 2 4 8 32"
          for card in ${CARD_LIST}
          do
            python3 ml_profiler.py -p ${PROFILE} -m ${model} -c ${card} -b ${batch} -o ${opt} -r ${lr} -e ${EPOCH}
          done
        fi
      done
    done
  done
done




