if [ $# == 0 ]
then
    echo "parameter error, usage: mem-util.sh [output_name]"
    echo "full cmd: mem-util.sh [output_name] [GPU_id] [img_width] [img_height]"
    exit 1
fi

datetime=$(date +%Y%m%d%H%M)
timestamp=$(date +%Y%m%d%H%M%S.%3N)

fileName=/home/ruiliu/Development/mtml-tf/mt-schedule/exp-result/$1 

if [ -f $fileName ]
then
    rm $fileName
fi

case $# in
    1)
    python3 benchmark_packed.py & 
    nvidia-smi --query-gpu=utilization.memory --format=csv --loop-ms=100 >> $fileName &
    wait -n
    sleep 5
    pkill -P $$
    ;;
    2)
    python3 benchmark_packed.py -g $2 &
    nvidia-smi --id=$2 --query-gpu=utilization.memory --format=csv --loop-ms=100 >> $fileName & 
    wait -n 
    sleep 5
    pkill -P $$
    ;;
    3)
    python3 benchmark_packed.py -g $2 -iw $3 &
    nvidia-smi --id=$2 --query-gpu=utilization.memory --format=csv --loop-ms=100 >> $fileName &
    wait -n
    sleep 5
    pkill -P $$
    ;;
    4)
    python3 benchmark_packed.py -g $2 -iw $3 -ih $4 &
    nvidia-smi --id=$2 --query-gpu=utilization.memory --format=csv --loop-ms=100 >> $fileName &
    wait -n
    sleep 5
    pkill -P $$
    ;;
esac


#python3 benchmark_packed.py & 
#nvidia-smi --query-gpu=utilization.memory --format=csv --loop-ms=100 >> /home/ruiliu/Development/mtml-tf/mt-schedule/exp-result/$1 &
#wait -n
#sleep 5
#pkill -P $$

#if [ $RET -eq 0 ]
#then
#    exit 0
#else
#    echo "return is wrong"
#    exit 1
#checkprocess $MYPID

#checkret()
# {
#    while true
#    do
#        if [ $1 != 100 ]
#        then
#            exit 0
#        fi
#    done
# }

#echo "Expertiment starts $timestamp" > /home/ruiliu/Development/mtml-tf/mt-schedule/mem-monitor-$datetime.csv
#python3 benchmark_packed.py & nvidia-smi --query-gpu=timestamp,utilization.memory --format=csv --loop-ms=100 >> /home/ruiliu/Development/mtml-tf/mt-schedule/mem-monitor-$datetime.csv

#RET=100
#checkret $RET & nvidia-smi --query-gpu=utilization.memory --format=csv --loop-ms=100 >> /home/ruiliu/Development/mtml-tf/mt-schedule/mem-monitor-$datetime.csv & python3 benchmark_packed.py && RET=$? 
