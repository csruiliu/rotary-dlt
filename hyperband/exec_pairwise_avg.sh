#!/bin/bash
echo 'start pairwising'
START=0
END=31

for i in $(seq $START $END); 
do 
    touch 'exp_'$i.txt 
    python3 clean_gpu_cache.py
    rm -rf __pycache__
    SNG_A1=`python3 pairwise_engine.py --single $i`

    python3 clean_gpu_cache.py
    rm -rf __pycache__
    SNG_A2=`python3 pairwise_engine.py --single $i`

    python3 clean_gpu_cache.py
    rm -rf __pycache__
    SNG_A3=`python3 pairwise_engine.py --single $i`  
    
    for j in $(seq $i $END);
    do
        python3 clean_gpu_cache.py
        rm -rf __pycache__
        SNG_B1=`python3 pairwise_engine.py --single $j`

        python3 clean_gpu_cache.py
        rm -rf __pycache__
        SNG_B2=`python3 pairwise_engine.py --single $j`

        python3 clean_gpu_cache.py
        rm -rf __pycache__
        SNG_B3=`python3 pairwise_engine.py --single $j`

        python3 clean_gpu_cache.py
        rm -rf __pycache__
        PACK_AB1=`python3 pairwise_engine.py --packmode --pack $i,$j`

        python3 clean_gpu_cache.py
        rm -rf __pycache__
        PACK_AB2=`python3 pairwise_engine.py --packmode --pack $i,$j`

        python3 clean_gpu_cache.py
        rm -rf __pycache__
        PACK_AB3=`python3 pairwise_engine.py --packmode --pack $i,$j` 
        echo "scale=4; (($SNG_A1+$SNG_A2+$SNG_A3)/3.0 + ($SNG_B1+$SNG_B2+$SNG_B3)/3.0 - ($PACK_AB1+$PACK_AB2+$PACK_AB3)/3.0)/(($SNG_A1+$SNG_A2+$SNG_A3)/3.0 + ($SNG_B1+$SNG_B2+$SNG_B3)/3.0)" | bc >> 'exp_'$i.txt
    done
done

