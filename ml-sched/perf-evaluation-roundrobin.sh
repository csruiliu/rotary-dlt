#!/bin/sh

JOB_LIST="20 40 60 80 100"
TIME_SLOT="48 60 72 84 96"
for jidx in ${JOB_LIST}
do
  rm -rf __pycache__
  python3 clean_gpu_cache.py
  python3 schedule_evaluation_roundrobin.py -j ${jidx} >> ./EXP-JOB${jidx}.txt
done

for tidx in ${TIME_SLOT}
do
  rm -rf __pycache__
  python3 clean_gpu_cache.py
  python3 schedule_evaluation_roundrobin.py -t ${tidx} >> ./EXP-TIME${tidx}.txt
done