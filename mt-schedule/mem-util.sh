datetime=$(date +%Y%m%d%H%M)
timestamp=$(date +%Y%m%d%H%M%S.%3N)
python3 benchmark_packed.py & 
nvidia-smi --query-gpu=utilization.memory --format=csv --loop-ms=100 >> /home/ruiliu/Development/mtml-tf/mt-schedule/mem-monitor-$datetime.csv & 
wait -n
sleep 5
pkill -P $$

#if [ $RET -eq 0 ]
#then
#    exit 0
#else
#    echo "return is wrong"
#    exit 1
#checkprocess $MYPID

#checkret()
#{
#    while true
#    do
#        if [ $1 != 100 ]
#        then
#            exit 0
#        fi
#    done
#}

#echo "Expertiment starts $timestamp" > /home/ruiliu/Development/mtml-tf/mt-schedule/mem-monitor-$datetime.csv
#python3 benchmark_packed.py & nvidia-smi --query-gpu=timestamp,utilization.memory --format=csv --loop-ms=100 >> /home/ruiliu/Development/mtml-tf/mt-schedule/mem-monitor-$datetime.csv

#RET=100
#checkret $RET & nvidia-smi --query-gpu=utilization.memory --format=csv --loop-ms=100 >> /home/ruiliu/Development/mtml-tf/mt-schedule/mem-monitor-$datetime.csv & python3 benchmark_packed.py && RET=$? 
