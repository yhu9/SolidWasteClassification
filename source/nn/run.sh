#!/bin/bash

for f in $(ls features_extracted)
do
    full_path="features_extracted/$f"
    #echo $full_path
    python3 nn.py train $full_path
    python3 nn.py train $full_path pca

    for img in $(ls ../categories/mixed)
    do
        full_imgpath="../categories/mixed/$img"
        tmp=$(echo $f | cut -c 10-)
        tmp1="${tmp%.*}"
        tmp2="_model"
        tmp2a="_pcamodel"
        tmp3="nn_model.ckpt"
        full_modelpath1=$tmp1a$tmp2/$tmp3
        full_modelpath2=$tmp1a$tmp2a/$tmp3
        #echo $full_imgpath $full_modelpath1
        #echo $full_imgpath $full_modelpath2 pca $full_path
        python3 nn.py test $full_imgpath $full_modelpath1
        python3 nn.py test $full_imgpath $full_modelpath2 pca $full_path
    done
done


