#!/bin/bash
echo 'start pairwising'
END=0
for i in $(seq 0 $END); 
do 
    touch 'exp_'$i.txt 
    SNG_A=`python3 pairwise_engine.py --single $i`  
    for j in $(seq $i $END);
    do
        SNG_B=`python3 pairwise_engine.py --single $j`
        PACK_AB=`python3 pairwise_engine.py --packmode --pack $i,$j` 
        
        SEQ=`$SNG_A + $SNG_B | bc`
        
        echo $SEQ
        #echo $((($SNG_B + $SNG_A)-$PACK_AB)/($SNG_A+$SNG_B))
    done
done