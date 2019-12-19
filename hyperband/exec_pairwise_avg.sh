#!/bin/bash
echo 'start pairwising'
START=0
END=1
EP=1

AVG_LIST=()
for i in $(seq $START $END);
do 
    TOTAL=0
    COUNT=0
    for j in $(seq 1 $EP);
    do 
        python3 clean_gpu_cache.py
        rm -rf __pycache__
        SNG=`python3 pairwise_engine.py --single $i`         
        TOTAL=$(echo $TOTAL+$SNG | bc )
        ((COUNT++))
    done 
    AVG=$(echo "scale=4; $TOTAL / $COUNT" | bc)
    AVG_LIST+=($AVG)    
done

for i in $(seq $START $END);
do
    for j in $(seq i $END);
    do  
        TOTAL_PACK=0
        COUNT_PACK=0
        for k in $(seq 1 $EP);
            python3 clean_gpu_cache.py
            rm -rf __pycache__
            PACK_AB=`python3 pairwise_engine.py --packmode --pack $i,$j`
            TOTAL_PACK=$(echo $TOTAL_PACK+$PACK_AB | bc )
            ((COUNT_PACK++))
        do
        done
        AVG_PACK=$(echo "scale=4; $TOTAL_PACK / $COUNT_PACK" | bc)
        touch 'exp_'$i.txt
        echo "scale=4; (${AVG_LIST[i]} + ${AVG_LIST[j]} - $AVG_PACK)/(${AVG_LIST[i]}+${AVG_LIST[j]})" | bc >> 'exp_'$i.txt
    done
done
