#!/bin/sh

JOB_LIST="20 50 80 110 140"
TIME_SLOT="48 60 72 84 96"
for jidx in ${JOB_LIST}
do
  rm -rf __pycache__
  python3 clean_gpu_cache.py
  python3 evaluation_roundrobin.py -j ${jidx} >> ./EXP-JOB${jidx}.txt
done

for tidx in ${TIME_SLOT}
do
  rm -rf __pycache__
  python3 clean_gpu_cache.py
  python3 evaluation_roundrobin.py -t ${tidx} >> ./EXP-TIME${tidx}.txt
done