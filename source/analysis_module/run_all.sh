#!/bin/bash


DIR1="../hsvsegmentation_bgrsegment_nobg"
DIR2="../hsvsegmentation_nobg"
DIR3="../bgrsegmentation_nobg"

items=(hog size gabor color hsv)
n=${#items[@]}
powersize=$((1 << $n))

i=0
while [ $i -lt $powersize ]
do
    subset=()
    j=0
    while [ $j -lt $n ]
    do
        if [ $(((1 << $j) & $i)) -gt 0 ]
        then
            subset+=("${items[$j]}")
        fi
        j=$(($j + 1))
    done

    argument="${subset[@]}"
    if [ -n "$argument" ]
    then
        python test.py $DIR1 $argument
        python test.py $DIR2 $argument
        python test.py $DIR3 $argument
    fi
    i=$(($i + 1))
done

wait

mkdir features_extracted
mv *features_*.txt features_extracted
mv features_extracted ../nn

echo "extracted all features"
