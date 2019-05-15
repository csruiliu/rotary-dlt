#!/bin/bash
cmd_arr=("pack_mm.py" "pack_rr.py" "pack_rm.py" "pack_io_mm.py" "pack_io_rr.py" "pack_io_rm.py")
para_arr=("10" "20" "40" "80" "100")
loop=("1" "2" "3")

for it in ${loop[@]}
do
  for cmd in ${cmd_arr[@]}
  do
    for para in ${para_arr[@]}
    do
      cplcmd="python "$cmd" -b "$para" | tee "$cmd$para"-"$it".txt"
      #echo $cplcmd
      eval $cplcmd
    done
  done
done
