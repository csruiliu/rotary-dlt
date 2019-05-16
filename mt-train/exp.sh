#!/bin/bash
cmd_arr=("pack_mm.py")
batch_arr=("10")
loop=("1")
eval="epoch"

for it in ${loop[@]}
do
  for cmd in ${cmd_arr[@]}
  do
    for para in ${batch_arr[@]}
    do
      cplcmd="python "$cmd" -b "$para" -e "$eval" | tee "$cmd$para$eval"-"$it".txt"
      eval $cplcmd
    done
  done
done
