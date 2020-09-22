#!/bin/sh

JOB_LIST="10 50 100 150 200"
TIME_SLOT="48 72 96 120"
for jidx in ${JOB_LIST}
do
  python3 schedule_evaluation_roundrobin.py -j ${jidx} >> ./EXP-JOB${jidx}.txt
done

rm -rf __pycache__
python3 clean_gpu_cache.py

for tidx in ${TIME_SLOT}
do
  python3 schedule_evaluation_roundrobin.py -t ${tidx} >> ./EXP-TIME${tidx}.txt
done