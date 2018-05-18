#!/bin/bash

size=200

'
cd cnn3

python3 cnn3.py test ../full_img.png model_big/cnn3_model.ckpt &&
mv results.txt results_big/result_big$size
mv learned_segmentation.png results_big/learned_segmentation$size.png
mv unlearned_segmentation.png results_big/unlearned_segmentation$size.png &&

python3 cnn3.py test ../full_img.png model_huge/cnn3_model.ckpt &&
mv results.txt results_huge/result_huge$size
mv learned_segmentation.png results_huge/learned_segmentation$size.png
mv unlearned_segmentation.png results_huge/unlearned_segmentation$size.png &&

python3 cnn3.py test ../full_img.png model_enormous/cnn3_model.ckpt &&
mv results.txt results_enormous/result_enormous$size
mv learned_segmentation.png results_enormous/learned_segmentation$size.png
mv unlearned_segmentation.png results_enormous/unlearned_segmentation$size.png &&

cd ..

cd cnn4

python3 cnn4.py test ../full_img.png model_big/cnn4_model.ckpt &&
mv results.txt results_big/result_big$size
mv learned_segmentation.png results_big/learned_segmentation$size.png
mv unlearned_segmentation.png results_big/unlearned_segmentation$size.png &&

python3 cnn4.py test ../full_img.png model_huge/cnn4_model.ckpt &&
mv results.txt results_huge/result_huge$size
mv learned_segmentation.png results_huge/learned_segmentation$size.png
mv unlearned_segmentation.png results_huge/unlearned_segmentation$size.png &&

python3 cnn4.py test ../full_img.png model_enormous/cnn4_model.ckpt &&
mv results.txt results_enormous/result_enormous$size
mv learned_segmentation.png results_enormous/learned_segmentation$size.png
mv unlearned_segmentation.png results_enormous/unlearned_segmentation$size.png &&

cd ..
'

cd cnn5

python3 cnn5.py test ../full_img.png model_big/cnn5_model.ckpt &&
mv results.txt results_big/result_big$size
mv learned_segmentation.png results_big/learned_segmentation$size.png
mv unlearned_segmentation.png results_big/unlearned_segmentation$size.png &&

python3 cnn5.py test ../full_img.png model_huge/cnn5_model.ckpt &&
mv results.txt results_huge/result_huge$size
mv learned_segmentation.png results_huge/learned_segmentation$size.png
mv unlearned_segmentation.png results_huge/unlearned_segmentation$size.png &&

python3 cnn5.py test ../full_img.png model_enormous/cnn5_model.ckpt &&
mv results.txt results_enormous/result_enormous$size
mv learned_segmentation.png results_enormous/learned_segmentation$size.png
mv unlearned_segmentation.png results_enormous/unlearned_segmentation$size.png &&

cd ..
