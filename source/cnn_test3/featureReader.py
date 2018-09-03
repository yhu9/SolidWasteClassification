import cv2
import numpy as np
import random
import os
import re
import math
import constants
import scipy.misc
from segmentModule import *
from matplotlib import pyplot as plt


#gets the full directory of the ground truth file for the corresponding img file
def getGTNameFromImgName(img_name):
    base = os.path.basename(img_name)
    split = os.path.splitext(base)
    fname = split[0] + '_gt.png'

    gtname = os.path.join(constants.GTDIR,fname)

    return gtname

#gets n patches from an image with its respective label
def getTrainingBatch(n,catname='mixed'):
    random.seed(None)
    inputs = np.empty((n,constants.IMG_SIZE,constants.IMG_SIZE,constants.IMG_DEPTH))
    labels = []

    #pick a random file to train on along with its ground truth file
    files = os.listdir(constants.MIXEDDIR)
    index = random.randint(0,len(files) - 1)
    full_dir = os.path.join(constants.MIXEDDIR,files[index])
    full_gtdir = getGTNameFromImgName(full_dir)

    #initialize variables
    tmp = cv2.imread(full_dir,cv2.IMREAD_COLOR)
    img = cv2.resize(tmp,(constants.FULL_IMGSIZE,constants.FULL_IMGSIZE),interpolation=cv2.INTER_CUBIC)
    tmp = cv2.imread(full_gtdir,cv2.IMREAD_COLOR)
    gt = cv2.resize(tmp,(constants.FULL_IMGSIZE,constants.FULL_IMGSIZE),interpolation=cv2.INTER_NEAREST)

    #preprocess label image to be of 2^3 color scheme
    gt[gt <= 128] = 0
    gt[gt > 128] = 255

    #check output format of model
    #category location based on output format
    if catname != 'mixed':
        binary = True
        onehotindex = constants.CATS.index(catname)
        gtcolor = constants.CATGT[onehotindex]
        truthmap = np.all(gt == gtcolor,axis=-1)
    else:
        binary = False
        h,w = gt.shape[:2]
        truthmap = np.full((h,w),True,dtype=np.bool)

    #get pixel batch but even out the pixels to include the different categories evenly
    w, h, d = img.shape
    j = 0
    while j < n:
        j += 1

        #get the points corresponding to the category we wish to train on
        if j % 2 == 0 or (not binary):
            rows,cols = np.where(truthmap == True)
        else:
            rows,cols = np.where(truthmap == False)

        #pick a random point on the image that is away from the edges
        tmp1 = np.logical_and(rows >= math.ceil(constants.IMG_SIZE / 2),rows < math.ceil(w - constants.IMG_SIZE / 2))
        tmp2 = np.logical_and(cols >= math.ceil(constants.IMG_SIZE / 2),cols < math.ceil(h - constants.IMG_SIZE / 2))
        tmpmap = np.logical_and(tmp1,tmp2)      #create a truth map
        idlist, = np.where(tmpmap == True)       #create a list where the truth map is true
        index = random.randint(0,len(idlist) - 1) #pick a random point on the list
        a = rows[idlist[index]]
        b = cols[idlist[index]]

        #find the label for the point
        if binary and ((j % 2) == 0):
            labels.append([1])
        elif binary and ((j % 2) == 1):
            labels.append([0])
        elif not binary:
            if np.all(gt[a,b] == [0,0,255]):
                labels.append(constants.CAT1_ONEHOT)
            elif np.all(gt[a,b] == [0,255,0]):
                labels.append(constants.CAT2_ONEHOT)
            elif np.all(gt[a,b] == [255,0,0]):
                labels.append(constants.CAT3_ONEHOT)
            elif np.all(gt[a,b] == [0,255,255]):
                labels.append(constants.CAT4_ONEHOT)
            elif np.all(gt[a,b] == [255,0,255]):
                labels.append(constants.CAT5_ONEHOT)
            elif np.all(gt[a,b] == [255,255,0]):
                labels.append(constants.CAT6_ONEHOT)
            else:
                #skip the point because its not one of the category colors
                j -=1
                continue

        #get the bounding box for the point
        low = int(constants.IMG_SIZE / 2)
        high = int(w - (constants.IMG_SIZE / 2) - 1)
        box_low1 = int(a - (constants.IMG_SIZE / 2))
        box_low2 = int(b - (constants.IMG_SIZE / 2))
        box_high1 = int(a + (constants.IMG_SIZE / 2))
        box_high2 = int(b + (constants.IMG_SIZE / 2))
        box = img[box_low1:box_high1,box_low2:box_high2]
        inputs[j-1] = box

    #return as batch size to get normal distribution
    return inputs,np.array(labels)

#gets n patches from an image with its respective label
def getTestingBatch(n,catname='mixed'):
    cats = ['treematter','plywood','cardboard','bottles','trashbag','blackbag']
    random.seed(None)
    inputs = []
    labels = []

    #initialize variables
    tmp = cv2.imread(constants.TESTFILE,cv2.IMREAD_COLOR)
    img = cv2.resize(tmp,(constants.FULL_IMGSIZE,constants.FULL_IMGSIZE),interpolation=cv2.INTER_CUBIC)
    tmp = cv2.imread(getGTNameFromImgName(constants.TESTFILE),cv2.IMREAD_COLOR)
    gt = cv2.resize(tmp,(constants.FULL_IMGSIZE,constants.FULL_IMGSIZE),interpolation=cv2.INTER_NEAREST)

    #preprocess label image to be of 2^3 color scheme
    gt[gt <= 128] = 0
    gt[gt > 128] = 255

    #get pixel batch
    w, h, d = img.shape
    j = 0
    while j < n:
        j += 1
        low = int(constants.IMG_SIZE / 2)
        high = int(w - (constants.IMG_SIZE / 2) - 1)
        a = random.randint(low,high)
        b = random.randint(low,high)
        box_low1 = int(a - (constants.IMG_SIZE / 2))
        box_low2 = int(b - (constants.IMG_SIZE / 2))
        box_high1 = int(a + (constants.IMG_SIZE / 2))
        box_high2 = int(b + (constants.IMG_SIZE / 2))

        #get label
        if np.all(gt[a,b] == [0,0,255]):
            labels.append(constants.CAT1_ONEHOT)
        elif np.all(gt[a,b] == [0,255,0]):
            labels.append(constants.CAT2_ONEHOT)
        elif np.all(gt[a,b] == [255,0,0]):
            labels.append(constants.CAT3_ONEHOT)
        elif np.all(gt[a,b] == [0,255,255]):
            labels.append(constants.CAT4_ONEHOT)
        elif np.all(gt[a,b] == [255,0,255]):
            labels.append(constants.CAT5_ONEHOT)
        elif np.all(gt[a,b] == [255,255,0]):
            labels.append(constants.CAT6_ONEHOT)
        else:
            j -= 1
            continue

        #get the box
        box = img[box_low1:box_high1,box_low2:box_high2,:]
        inputs.append(box)

    #shuffle the input and labels in parralel
    c = list(zip(inputs,labels))
    random.shuffle(c)
    inputs,labels = zip(*c)
    inputs = np.array(inputs)[:n,:,:,:]
    labels = np.array(labels)[:n]
    cat1label = np.zeros(len(labels))
    cat2label = np.zeros(len(labels))
    cat3label = np.zeros(len(labels))
    cat4label = np.zeros(len(labels))
    cat5label = np.zeros(len(labels))
    cat6label = np.zeros(len(labels))
    cat1label[np.all(labels == constants.CAT1_ONEHOT) == 1] = 1
    cat2label[np.all(labels == constants.CAT2_ONEHOT) == 1] = 1
    cat3label[np.all(labels == constants.CAT3_ONEHOT) == 1] = 1
    cat4label[np.all(labels == constants.CAT4_ONEHOT) == 1] = 1
    cat5label[np.all(labels == constants.CAT5_ONEHOT) == 1] = 1
    cat6label[np.all(labels == constants.CAT6_ONEHOT) == 1] = 1

    #return as batch size to get normal distribution
    return inputs,labels,cat1label.reshape((n,1)),cat2label.reshape((n,1)),cat3label.reshape((n,1)),cat4label.reshape((n,1)),cat5label.reshape((n,1)),cat6label.reshape((n,1))

