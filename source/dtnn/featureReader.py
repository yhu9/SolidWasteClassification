import cv2
import numpy as np
import random
import os
import re
import math
import constants
import scipy.misc
import extractionModule as analyze
import sys
from sklearn.decomposition import PCA
from segmentModule import *
from matplotlib import pyplot as plt

#generates the instances and labels for all instances within a text file
'''
INPUTS:
    1. text file containing each instance line by line and the label marked with a #
OUTPUTS:
    1. instances as 2d numpy array
    2. labels for each instance as 1d numpy array
'''
def genFromText(filepath):

    labels = []
    instances = np.genfromtxt(filepath,delimiter=',',dtype=np.float)

    tmp  = np.load(filepath)
    instances = tmp[:,:-1].astype(float)
    labels = tmp[:,-1:].astype(str)
    categories = np.zeros((labels.shape[0],labels.shape[1],constants.NN_CLASSES))
    categories[labels == 'treematter'] = constants.CAT1_ONEHOT
    categories[labels == 'plywood'] = constants.CAT2_ONEHOT
    categories[labels == 'cardboard'] = constants.CAT3_ONEHOT
    categories[labels == 'bottles'] = constants.CAT4_ONEHOT
    categories[labels == 'trashbag'] = constants.CAT5_ONEHOT
    categories[labels == 'blackbag'] = constants.CAT6_ONEHOT
    categories[labels == 'mixed'] = [0,0,0,0,0,0]

    #return the created content
    return instances,categories

#inputs are 1d extracted feature vectors
#labels are one hot
#returns inputs and labels
def getBatch(n,instances,labels):

    if n >= len(instances):
        n = len(instances) - 1
        print("batch size is too large for the instances given: %i" % len(instances))
        sys.exit()

    batch = []
    batch_labels = []

    #get a good mix
    cat1ids, = np.where(np.all(labels == [1,0,0,0,0,0],axis = 1))
    cat2ids, = np.where(np.all(labels == [0,1,0,0,0,0],axis = 1))
    cat3ids, = np.where(np.all(labels == [0,0,1,0,0,0],axis = 1))
    cat4ids, = np.where(np.all(labels == [0,0,0,1,0,0],axis = 1))
    cat5ids, = np.where(np.all(labels == [0,0,0,0,1,0],axis = 1))
    cat6ids, = np.where(np.all(labels == [0,0,0,0,0,1],axis = 1))

    for i in range(n):
        #roll a single dice
        roll_dice = random.randint(0,5)
        if roll_dice % 5 == 0:
            index = random.randint(0,(len(cat1ids) - 1))
        elif roll_dice % 5 == 1:
            index = random.randint(0,(len(cat2ids) - 1))
        elif roll_dice % 5 == 2:
            index = random.randint(0,(len(cat3ids) - 1))
        elif roll_dice % 5 == 3:
            index = random.randint(0,(len(cat4ids) - 1))
        elif roll_dice % 5 == 4:
            index = random.randint(0,(len(cat5ids) - 1))
        elif roll_dice % 5 == 5:
            index = random.randint(0,(len(cat6ids) - 1))

        batch.append(instances[index])
        batch_labels.append(labels[index])

    #return the created content
    return np.array(batch),np.array(batch_labels)


#extracts blobs from image using the meanshift algorithm
'''
INPUTS:
    1. image
OUTPUTS:
    1. 3d numpy array with blobs as instances as 2d numpy arrays
    2. 1d numpy array as labels
'''
def extractBlobs(img,fout="unsupervised_segmentation.png",hsv=False):
    blobs = []
    if hsv:
        hsvimg = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        original,markers = getSegments(hsvimg,False)
    else:
        original,markers = getSegments(img,False)

    labels = np.unique(markers)
    canvas = original.copy()
    for uq_mark in labels:
        #get the segment and append it to inputs
        region = original.copy()
        region[markers != uq_mark] = [0,0,0]
        grey = cv2.cvtColor(region,cv2.COLOR_BGR2GRAY)
        x,y,w,h = cv2.boundingRect(grey)
        cropped = img[y:y+h,x:x+w]
        cropped = np.uint8(cropped)
        blobs.append(cropped)

        b = random.randint(0,255)
        g = random.randint(0,255)
        r = random.randint(0,255)
        canvas[markers == uq_mark] = [b,g,r]

    cv2.imwrite(fout,canvas)
    print("Unsupervised segmentation saved! %s" % fout)

    return blobs, markers, labels

#creates the testing instances from the image by extracting the blobs and evaluating hog/color/gabor features
'''
INPUTS:
    1. cv2 image
OUTPUTS:
    1. 2d numpy array as feature vector instances
    2. 2d numpy array as a mask for the image
    3. 1d numpy array as labels for each instance
'''
def createTestingInstancesFromImage(image,hsvseg=False,hog=False,color=False,gabor=False,size=False,hsv=False,filename="unsupervised_segmentation.png"):
    #segment the image and extract blobs
    print("extracting blobs")
    blobs,markers,labels = extractBlobs(image,fout=filename,hsv=hsvseg)

    #evaluate each blob and extract features from them
    tmp = []
    for i,blob in enumerate(blobs):
        featurevector = analyze.evaluateSegment(blob,hogflag=hog,colorflag=color,gaborflag=gabor,sizeflag=size,hsvflag=hsv)
        #console output to show progress
        print("%i of blob %i ---> FEATURES EXTRACTED" % (i + 1, len(blobs)))
        tmp.append(featurevector)

    #create numpy array
    instances = np.array(tmp)

    #normalize the sizes across all blobs
    if size:
        instances[:,0] = analyze.normalize(instances[:,0])

    return instances, markers, labels

#given an image and its mask writes the results as fout
def outputResults(image,mask,fout='segmentation.png'):
    #create the segmented image
    canvas = image.copy()
    canvas[mask == -1] = [0,0,0]
    canvas[mask == 0] = [0,0,255]
    canvas[mask == 1] = [0,255,0]
    canvas[mask == 2] = [255,0,0]
    canvas[mask == 3] = [0,255,255]
    canvas[mask == 4] = [255,0,255]
    canvas[mask == 5] = [255,255,0]

    #show the original image and the segmented image and then save the results
    cv2.imwrite(fout,canvas)

    #count the percentage of each category
    cat0_count = np.count_nonzero(mask == -1)
    cat1_count = np.count_nonzero(mask == 0)
    cat2_count = np.count_nonzero(mask == 1)
    cat3_count = np.count_nonzero(mask == 2)
    cat4_count = np.count_nonzero(mask == 3)
    cat5_count = np.count_nonzero(mask == 4)
    cat6_count = np.count_nonzero(mask == 5)
    total = cat1_count + cat2_count + cat3_count + cat4_count + cat5_count + cat6_count + cat0_count

    #get the percentage of each category
    p1 = cat1_count / total
    p2 = cat2_count / total
    p3 = cat3_count / total
    p4 = cat4_count / total
    p5 = cat5_count / total
    p6 = cat6_count / total

    #output to text file
    with open('results.txt','a') as f:
        f.write("\nusing model: %s\n" % sys.argv[3])
        f.write("evaluate image: %s\n\n" % sys.argv[2])
        f.write("--------------------------------------------------------------------------------------\n")
        f.write("%s : %f\n" % (constants.CAT1,p1))
        f.write("%s : %f\n" % (constants.CAT2,p2))
        f.write("%s : %f\n" % (constants.CAT3,p3))
        f.write("%s : %f\n" % (constants.CAT4,p4))
        f.write("%s : %f\n" % (constants.CAT5,p5))
        f.write("%s : %f\n" % (constants.CAT6,p6))
        f.write("--------------------------------------------------------------------------------------\n")
        f.write("------------------------------------END-----------------------------------------------\n")
        f.write("--------------------------------------------------------------------------------------\n")

        greatest = max(cat1_count,cat2_count,cat3_count,cat4_count)

        #f.write out to the terminal what the most common category was for the image
        if(greatest == cat1_count):
            f.write("\nthe most common category is: " + constants.CAT1)
        elif(greatest == cat2_count):
            f.write("\nthe most common category is: " + constants.CAT2)
        elif(greatest == cat3_count):
            f.write("\nthe most common category is: " + constants.CAT3)
        elif(greatest == cat4_count):
            f.write("\nthe most common category is: " + constants.CAT4)
        elif(greatest == cat5_count):
            f.write("\nthe most common category is: " + constants.CAT5)
        elif(greatest == cat6_count):
            f.write("\nthe most common category is: " + constants.CAT6)
        else:
            f.write("\nsorry something went wrong counting the predictions")

#applies the pca singular values on a feature vector and returns the results
def applyPCA(pca,inputfeatures,log_dir='dump.txt'):

    if len(inputfeatures.shape) == 1:
        newfeatures = pca.transform(inputfeatures.reshape(1,len(inputfeatures)))
    else:
        newfeatures = pca.transform(inputfeatures)

    with open(log_dir,'a') as fout:
        fout.write("\n1. -------------------------------------------------------------------------------------------\n")
        fout.write(str(pca.n_components_))
        fout.write("\n\n")
        fout.write("2. -------------------------------------------------------------------------------------------\n")
        fout.write(str(pca.explained_variance_ratio_))
        fout.write("\n\n")
        fout.write("3. -------------------------------------------------------------------------------------------\n")
        fout.write("feature length: " + str(len(inputfeatures[0])) + "      reduced length: " + str(len(newfeatures[0])))
        fout.write("\n\n")

    return newfeatures

#get the PCA analysis and fit it to the featurevector of instances
def getPCA(featurevector,featurelength=constants.PCA_LENGTH):

    #http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    #all default values except for n_components
    pca = PCA(copy=True, iterated_power='auto', n_components=constants.PCA_LENGTH, random_state=None,
              svd_solver='auto', tol=0.0, whiten=False)

    #apply pca on the feature vector
    pca.fit(featurevector)

    return pca


