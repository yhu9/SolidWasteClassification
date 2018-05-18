#!/bin/bash

python3 nn.py train features_hsvsegmentation_bgrsegments.txt &&
python3 nn.py train features_bgr_segments_nobg.txt &&
python3 nn.py train features_hsv_segments_nobg.txt &&

python3 nn.py trainpca features_hsvsegmentation_bgrsegments.txt -n 80 &&
python3 nn.py trainpca features_bgr_segments_nobg.txt -n 75 &&
python3 nn.py trainpca features_hsv_segments_nobg.txt -n 111

