#datetime=$(date +%Y%m%d%H%M)
#timestamp=$(date +%Y%m%d%H%M%S.%3N)

#Query Type can be one of the following:
#utilization.gpu
#utilization.memory
#memory.total
#memory.free
#memory.used

PYCMD=""
NVISMI=""
lms=10
queryType="memory.gpu"
while getopts "f:q:g:e:b:w:h:sdl:" opt; do
    case $opt in
        f)        
        fileName=$OPTARG
        ;;
        q)
        queryType=$OPTARG
        ;;
        g)
        PYCMD+=" -g $OPTARG"
        ;;
        e)
        PYCMD+=" -e $OPTARG"
        ;;
        b)
        PYCMD+=" -bs $OPTARG"
        ;;
        w)
        PYCMD+=" -iw $OPTARG"
        ;;
        h)
        PYCMD+=" -ih $OPTARG"
        ;;
        s)
        PYCMD+=" -s"
        ;;
        d)
        PYCMD+=" -d"
        ;;
        p)
        PYCMD+=" -p"
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

#echo $PYCMD

python3 perf_sch.py$PYCMD &
nvidia-smi --query-gpu=$queryType --format=csv --loop-ms=$lms >> $FilePath &
wait -n
sleep 5
pkill -P $$
