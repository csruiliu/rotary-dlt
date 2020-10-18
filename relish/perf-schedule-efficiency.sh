#!/bin/bash

FILE_LIST="exp-roundrobin-time60 exp-roundrobin-time72 exp-roundrobin-time84 exp-roundrobin-time96"

for fidx in ${FILE_LIST}
do
  python3 profiler_train_time_isolate.py -p ${fidx}.txt >> output-${fidx}
done
