#datetime=$(date +%Y%m%d%H%M)
#timestamp=$(date +%Y%m%d%H%M%S.%3N)

PYCMD=""
while getopts "f:g:e:w:h:s" opt; do
    case $opt in
        f)
        fileName=$OPTARG
        ;;
        g)
        pycmd+=" -g $OPTARG"
        ;;
        e)
        pycmd+=" -e $OPTARG"
        ;;
        w)
        pycmd+=" -iw $OPTARG"
        ;;
        h)
        pycmd+=" -ih $OPTARG"
        ;;
        s)
        pycmd+=" -s"
    esac
done

FilePath=/home/ruiliu/Development/mtml-tf/mt-perf/exp-result/$fileName 

if [ -f $FilePath ]
then
    echo "log file exists, removing it"
    rm $FilePath
fi

python3 benchmark_packed.py$PYCMD &
nvidia-smi --query-gpu=utilization.memory --format=csv --loop-ms=100 >> $FilePath &
wait -n
sleep 5
pkill -P $$