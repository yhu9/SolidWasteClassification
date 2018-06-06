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

#reads in training image for cnn using pixel data as the training set
#28 x 28 surrounding area of each pixel used for training
#3x3 conv, 7x7 conv
#all training images must be passed when calling nn.py

def cnn_readOneImg2(image_dir):
    inputs = []
    img = cv2.imread(image_dir,cv2.IMREAD_COLOR)
    original,markers = getSegments(img,False)
    uniqueMarkers = np.unique(markers)
    canvas = original.copy()
    for uq_mark in uniqueMarkers:
        #make a canvas and paint each unique segment
        b = random.randint(0,255)
        g = random.randint(0,255)
        r = random.randint(0,255)
        canvas[markers == uq_mark] = [b,g,r]

    return(canvas,markers)

#function for stitching 4 different images together
#top left = cat1 = 'treematter'
#top right = cat2 = 'plywood'
#bot left = cat3 = 'cardboard'
#bot right = cat4 = 'construction'
def getStitchedImage():

    #initialize variablees
    cat1_dir = constants.cat1_dir
    cat2_dir = constants.cat2_dir
    cat3_dir = constants.cat3_dir
    cat4_dir = constants.cat4_dir
    cat1files = []
    cat2files = []
    cat3files = []
    cat4files = []

    #check if the file directories exist and push all files into their respective categories
    if os.path.exists(cat1_dir) and os.path.exists(cat2_dir) and os.path.exists(cat3_dir) and os.path.exists(cat4_dir):
        for filename in os.listdir(cat1_dir):
            cat1files.append(filename)
        for filename in os.listdir(cat2_dir):
            cat2files.append(filename)
        for filename in os.listdir(cat3_dir):
            cat3files.append(filename)
        for filename in os.listdir(cat4_dir):
            cat4files.append(filename)

    #pick a random file from the list of files for each category and read them in
    random.seed(None)
    a = random.randint(0,len(cat1files) - 1)
    b = random.randint(0,len(cat2files) - 1)
    c = random.randint(0,len(cat3files) - 1)
    d = random.randint(0,len(cat4files) - 1)
    img1 = cv2.imread(cat1_dir + '/' + cat1files[a],cv2.IMREAD_COLOR)
    img2 = cv2.imread(cat2_dir + '/' + cat2files[b],cv2.IMREAD_COLOR)
    img3 = cv2.imread(cat3_dir + '/' + cat3files[c],cv2.IMREAD_COLOR)
    img4 = cv2.imread(cat4_dir + '/' + cat4files[d],cv2.IMREAD_COLOR)

    #create the image by resizing and putting them into their correct positions
    topleft = cv2.resize(img1,(500,500),interpolation = cv2.INTER_CUBIC)
    bottomleft = cv2.resize(img2,(500,500),interpolation = cv2.INTER_CUBIC)
    topright = cv2.resize(img3,(500,500),interpolation = cv2.INTER_CUBIC)
    bottomright = cv2.resize(img4,(500,500),interpolation = cv2.INTER_CUBIC)
    toprow = np.concatenate((topleft,topright),axis = 1)
    bottomrow = np.concatenate((bottomleft,bottomright),axis = 1)
    full_img = np.concatenate((toprow,bottomrow),axis = 0)

    return full_img

def testStitcher():
    for i in range(10):
        full_img = stitchImage()
        rgb = scipy.misc.toimage(full_img)
        cv2.imshow('stiched image',full_img)
        cv2.imwrite('full_img.png',full_img)
        cv2.waitKey(0)

#gets n patches from an image with its respective label
def getPixelBatch(n):
    inputs = []
    labels = []

    #initialize variablees
    cat1_dir = constants.cat1_dir
    cat2_dir = constants.cat2_dir
    cat3_dir = constants.cat3_dir
    cat4_dir = constants.cat4_dir
    cat5_dir = constants.cat5_dir
    cat6_dir = constants.cat6_dir
    dirs = [constants.cat1_dir,constants.cat2_dir,constants.cat3_dir,constants.cat4_dir,constants.cat5_dir,constants.cat6_dir]
    categories = [constants.CAT1_ONEHOT,constants.CAT2_ONEHOT,constants.CAT3_ONEHOT,constants.CAT4_ONEHOT,constants.CAT5_ONEHOT,constants.CAT6_ONEHOT]
    images = []
    files = []

    #check if the file directories exist and push all files into their respective categories
    for d in dirs:
        if os.path.exists(d):
            files.append([])
            for fname in os.listdir(d):
                files[-1].append(fname)
        else:
            print("%s directory does not exist" % d)

    #pick a random file from the list of files for each category and read them in
    random.seed(None)
    for i,f_list in enumerate(files):
        a = random.randint(0,len(f_list) - 1)
        img1 = cv2.imread(dirs[i] + '/' + f_list[a],cv2.IMREAD_COLOR)
        images.append(cv2.resize(img1,(constants.FULL_IMGSIZE,constants.FULL_IMGSIZE),interpolation = cv2.INTER_CUBIC))


    if(len(images) == len(categories)):
        for img,cat in zip(images,categories):
            w, h, d = img.shape
            for j in range(int(n / 4)):
                low = int(constants.IMG_SIZE / 2)
                high = int(w - (constants.IMG_SIZE / 2) - 1)
                a = random.randint(low,high)
                b = random.randint(low,high)
                box_low1 = int(a - (constants.IMG_SIZE / 2))
                box_low2 = int(b - (constants.IMG_SIZE / 2))
                box_high1 = int(a + (constants.IMG_SIZE / 2))
                box_high2 = int(b + (constants.IMG_SIZE / 2))

                box = img[box_low1:box_high1,box_low2:box_high2]
                inputs.append(box)
                labels.append(cat)
    else:
        print('COULD NOT CREATE BATCH')
        print('invalid number of categories and images found: %i <--> %i' % (len(images),len(categories)))

    c = list(zip(inputs,labels))
    random.shuffle(c)
    inputs,labels = zip(*c)

    return inputs,labels

#get the batch of segments to process
def getSegmentBatch(n,filepath):

    #initialize variables
    seg_dir = filepath
    seg_fnames = os.listdir(seg_dir)
    inputs = []
    labels = []

    #start a random seed for the random number generator
    random.seed(None)
    for i in range(n):
        #get a random segment from list of segments
        rand_int = random.randint(0,len(seg_fnames) - 1)
        segment_file = seg_fnames[rand_int]
        label = re.findall("treematter|plywood|cardboard|bottles|trashbag|blackbag",segment_file)

        #push the found category into the labels list
        if(label[0] == constants.CAT1):
            labels.append(constants.CAT1_ONEHOT)
        elif(label[0] == constants.CAT2):
            labels.append(constants.CAT2_ONEHOT)
        elif(label[0] == constants.CAT3):
            labels.append(constants.CAT3_ONEHOT)
        elif(label[0] == constants.CAT4):
            labels.append(constants.CAT4_ONEHOT)
        elif(label[0] == constants.CAT5):
            labels.append(constants.CAT5_ONEHOT)
        elif(label[0] == constants.CAT6):
            labels.append(constants.CAT6_ONEHOT)
        else:
            print(label[0])

        #read the image and push it into the inputs list we have to normalize the image to a smaller pixel size.
        #we actually lose a ton of image quality here
        full_dir = seg_dir + segment_file
        img = cv2.imread(full_dir,cv2.IMREAD_COLOR)
        normal_img = cv2.resize(img,(constants.IMG_SIZE, constants.IMG_SIZE), interpolation = cv2.INTER_CUBIC)

        inputs.append(normal_img)

    return np.array(inputs),np.array(labels)

