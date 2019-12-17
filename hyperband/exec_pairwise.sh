#!/bin/bash
echo 'start pairwising'
END=31
for i in $(seq 0 $END); 
do 
    touch 'exp_'$i.txt 
    SNG_A=`python3 pairwise_engine.py --single $i`  
    `python3 clean_clean_gpu_cache.py`
    for j in $(seq $i $END);
    do
        SNG_B=`python3 pairwise_engine.py --single $j`
        `python3 clean_clean_gpu_cache.py`
        PACK_AB=`python3 pairwise_engine.py --packmode --pack $i,$j` 
        `python3 clean_clean_gpu_cache.py`
        #SEQ=`$SNG_A + $SNG_B | bc`
        #echo $SEQ
        echo "scale=4; ($SNG_A+$SNG_B-$PACK_AB)/($SNG_A+$SNG_B)" | bc >> 'exp_'$i.txt 
    done
done
