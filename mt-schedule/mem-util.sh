date +%Y%m%d%H%M%S.%3N > /home/ruiliu/Development/mtml-tf/mt-schedule/mem-util.csv
nvidia-smi --query-gpu=timestamp,utilization.memory --format=csv --loop-ms=100 >> /home/ruiliu/Development/mtml-tf/mt-schedule/mem-util.csv
