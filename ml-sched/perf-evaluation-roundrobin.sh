#!/bin/sh

JOB_LIST="50 100 150 200"

for jidx in ${JOB_LIST}
do
  python3 schedule_evaluation_roundrobin.py -n ${jidx} >> ./$EXP-JOB${jidx}.txt
done