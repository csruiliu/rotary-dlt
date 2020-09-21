#!/bin/sh

JOB_LIST="50 100 150 200"
TIME_SLOT="24 48 72 96"
for tidx in ${TIME_SLOT}
do
  python3 schedule_evaluation_roundrobin.py -n ${tidx} >> ./EXP-TIME${tidx}.txt
done