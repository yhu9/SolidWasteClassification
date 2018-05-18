#!/bin/bash

cd two-path &&
python3 two-path.py trainseg &&
mv model segmodel && mkdir model
mv two-path_log.txt results/two-paths_seglog.txt &&
python3 two-path.py trainpix &&
mv model pixmodel && mkdir model
mv two-path_log.txt results/two-paths_pixlog.txt &&
cd .. &&

cd cnn &&
python3 cnn.py trainseg &&
mv model segmodel && mkdir model
mv cnn_log.txt results/cnn_seglog.txt &&
python3 cnn.py trainpix &&
mv model pixmodel &&
mv cnn_log.txt results/cnn_pixlog.txt &&
cd .. &&

cd dcnn &&
python3 dcnn.py trainseg &&
mv model segmodel && mkdir model
mv dcnn_log.txt results/dcnn_seglog.txt &&
python3 dcnn.py trainpix &&
mv model pixmodel &&
mv dcnn_log.txt results/dcnn_pixlog.txt &&
cd .. &&
