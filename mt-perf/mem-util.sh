#datetime=$(date +%Y%m%d%H%M)
#timestamp=$(date +%Y%m%d%H%M%S.%3N)

#Query Type can be one of the following:
#utilization.gpu
#utilization.memory
#memory.total
#memory.free
#memory.used


PYCMD=""
while getopts "f:q:g:e:w:h:sdl:" opt; do
    case $opt in
        f)
        fileName=$OPTARG
        ;;
        q)
        queryType=$OPTARG
        ;;
        g)
        pycmd+=" -g $OPTARG"
        ;;
        e)
        pycmd+=" -e $OPTARG"
        ;;
        b)
        pycmd+=" -bs $OPTARG"
        ;;
        w)
        pycmd+=" -iw $OPTARG"
        ;;
        h)
        pycmd+=" -ih $OPTARG"
        ;;
        s)
        pycmd+=" -s"
        ;;
        d)
        pycmd+=" -d"
        ;;
        l)
        lms=$OPTARG
        ;;
    esac
done

#tank is a local disk of the server
#FilePath=/home/ruiliu/Development/mtml-tf/mt-perf/exp-result/$fileName 
FilePath=/tank/local/ruiliu/mtml-tf/mt-perf/exp-result/$fileName

if [ -f $FilePath ]
then
    echo "log file exists, removing it"
    rm $FilePath
fi

python3 perf_packed.py$PYCMD &
nvidia-smi --query-gpu=$queryType --format=csv --loop-ms=$lms >> $FilePath &
wait -n
sleep 5
pkill -P $$
