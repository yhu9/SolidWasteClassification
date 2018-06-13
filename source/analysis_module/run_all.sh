#!/bin/bash


DIR1="../hsvsegmentation_bgrsegment_nobg"
DIR2="../hsvsegmentation_nobg"
DIR3="../bgrsegmentation_nobg"


python test.py $DIR1 hog size gabor color hsv
python test.py $DIR2 hog size gabor color hsv
python test.py $DIR3 hog size gabor color hsv

wait

mkdir features_extracted
mv *features_*.npy features_extracted

echo "extracted all features"
