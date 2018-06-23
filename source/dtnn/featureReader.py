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
import pickle
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
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

    tmp  = np.load(filepath)
    instances = tmp[:,:-1].astype(float)
    labels = tmp[:,-1:].astype(np.int8)
    categories = np.zeros((labels.shape[0],1,constants.NN_CLASSES))
    categories[labels == 0] = constants.CAT1_ONEHOT     #Treematter
    categories[labels == 1] = constants.CAT2_ONEHOT     #plywood
    categories[labels == 2] = constants.CAT3_ONEHOT     #cardboard
    categories[labels == 3] = constants.CAT4_ONEHOT     #bottles
    categories[labels == 4] = constants.CAT5_ONEHOT     #trashbag
    categories[labels == 5] = constants.CAT6_ONEHOT     #blackbag
    categories[labels == -1] = [0,0,0,0,0,0]            #mixed

    #return the created content
    return instances,categories.reshape(labels.shape[0],constants.NN_CLASSES)

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
    print("finding blobs")
    blobs,markers,labels = extractBlobs(image,fout=filename,hsv=hsvseg)
    print("BLOBS FOUND!")
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
        instances[:,-1] = analyze.normalize(instances[:,-1])

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

#get the LDA analysis and fit it to the featurevector of instances
def getLDA(featurevector,labels):
    lda_labels = np.empty((labels.shape[0]))
    lda_labels[np.all(labels == [1,0,0,0,0,0],axis=1) == 1] = 0
    lda_labels[np.all(labels == [0,1,0,0,0,0],axis=1) == 1] = 1
    lda_labels[np.all(labels == [0,0,1,0,0,0],axis=1) == 1] = 2
    lda_labels[np.all(labels == [0,0,0,1,0,0],axis=1) == 1] = 3
    lda_labels[np.all(labels == [0,0,0,0,1,0],axis=1) == 1] = 4
    lda_labels[np.all(labels == [0,0,0,0,0,1],axis=1) == 1] = 5
    lda_labels[np.all(labels == [0,0,0,0,0,0],axis=1) == 1] = -1

    #all default values except for n_components
    lda = LDA()
    lda.fit(featurevector[lda_labels >= 0],lda_labels[lda_labels >= 0])

    return lda

#get the PCA analysis and fit it to the featurevector of instances
def getPCA(featurevector,featurelength=constants.DECOMP_LENGTH):

    #http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    #all default values except for n_components
    pca = PCA(copy=True, iterated_power='auto', n_components=featurelength, random_state=None,
              svd_solver='auto', tol=0.0, whiten=False)

    #apply pca on the feature vector
    newfeatures = pca.fit(featurevector)

    return pca

#applies the pca singular values on a feature vector and returns the results
def applyPCA(pca,inputfeatures):

    if len(inputfeatures.shape) == 1:
        newfeatures = pca.transform(inputfeatures.reshape(1,len(inputfeatures)))
    else:
        print(inputfeatures.shape)
        newfeatures = pca.transform(inputfeatures)

    return newfeatures


def applyLDA(lda,inputfeatures):
    if len(inputfeatures.shape) == 1:
        newfeatures = lda.transform(inputfeatures.reshape(1,len(inputfeatures)))
    else:
        newfeatures = lda.transform(inputfeatures)

    print('LDA applied to %i categories' % len(lda.classes_))

    return newfeatures

def loadPCA(filename):
    with open(filename,'rb') as f:
        pca = pickle.load(f,encoding='latin1')

    return pca

def loadLDA(filename):
    with open(filename,'rb') as f:
        lda = pickle.load(f,encoding='latin1')

    return lda


if __name__ == '__main__':

    if len(sys.argv) == 3:
        if sys.argv[1] == 'pca':
            featurefile = sys.argv[2]
            features,labels = genFromText(featurefile)
            pca = getPCA(features)
            pickle.dump(pca,open(os.path.splitext(os.path.basename(featurefile))[0] + '_pca.pkl','wb'))

        elif sys.argv[1] == 'lda':
            featurefile = sys.argv[2]
            features,labels = genFromText(featurefile)
            lda = getLDA(features,labels)
            pickle.dump(lda,open(os.path.splitext(os.path.basename(featurefile))[0] + '_lda.pkl','wb'))




