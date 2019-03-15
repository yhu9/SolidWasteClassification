#########################################################################################
Masa Hu, Dr. Martin Cenek
3-15-2019

SOLID WASTE CLASSIFICATION
#########################################################################################

IMPLEMENTATION OVERVIEW

The trash classification project looks to do image segmentation based on 6 classification categories (treematter,plywood,cardboard,blackbags,trashbags,plastic bottles). There are only a could ground truth images due to the difficulty in creating them. All data can be found on the google drive. The procedure is as followed:

Training:
1. Given a mixed image with the 6 categories
2. perform per pixel image classification by randomly cropping 32x32 squares at random locations on the mixed image
3. calculate loss as onehot classification error between ground truth
4. repeat until convergence of CNN model

Testing:
1. Given a mixed image with the 6 categories
2. Perform unsupervised meanshift segmentation to create blobs
3. use the trained model to infer the classification of each pixel in the image
4. save raw values to determine threshold as median of the raw values
5. eliminate pixel classifications which are below threshold value
6. determine blob classification by using majority class of each blob
7. output accuracy for each class

---------------------------------------------------------------------------------------------------
RESULTS FILES:

1 --> meanshift_seg
2 --> accuracy
3 --> learned_segmentation
4 --> threshmask
5 --> thresholds
6 --> majority_seg

1. Holds the unsupervised meanshift segmentation of the trash
2. Holds the accuracy results of classification for each category on the mixed image
3. Perpixel classification without thresholding
4. Perpixel classification with thresholding
5. threshold values for each image
6. Final segmentation based on majorit class of each blob


