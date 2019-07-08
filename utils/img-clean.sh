#!/bin/bash

imgStat="/home/ruiliu/Development/mtml-tf/utils/dirty.txt"
imgDir="/home/ruiliu/Development/mtml-tf/dataset/imagenet150k"
labelFile="/home/ruiliu/Development/mtml-tf/dataset/imagenet150k-label.txt"
testFile="/home/ruiliu/Development/mtml-tf/dataset/bak.txt"
imgIdx=149999
reg="000000"
while IFS= read -r line
do
    #echo "use ILSVRC2010_test_00$imgIdx.JPEG to replace $line"
    mv "$imgDir/ILSVRC2010_test_00$imgIdx.JPEG" "$imgDir/$line"

    s1=${line%'.'*}
    s2=${s1##*'_'} 
    s3=$reg$s2
    label_dirty=${s3##*'00000'}
    #echo $label_dirty
    label_replace=`sed -n $imgIdx',1p' $labelFile`
    echo $label_replace

    sed -i $label_dirty'c '$label_replace $testFile
    sed -i $imgIdx'd' $testFile

    #echo $label_replace
    
    #echo "$imgDir/$line"
    #echo "use ILSVRC2010_test_00$imgIdx.JPEG to replace $line"
    imgIdx=$(($imgIdx-1))
done < "$imgStat"
