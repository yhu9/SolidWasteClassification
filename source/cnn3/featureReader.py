import cv2
import numpy as np
import random
import os
import re
import math
import constants
import scipy.misc
import pywt
from skimage.feature import hog
from skimage import exposure
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

#gets n patches from an image with its respective label
def getBatch(n,catname='mixed'):
    cats = ['treematter','plywood','cardboard','bottles','trashbag','blackbag']
    random.seed(None)
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
            tmp = os.listdir(d)
            a = random.randint(0,len(tmp) - 1)
            path = os.path.join(d,tmp[a])
            files.append(path)
        else:
            print("%s directory does not exist" % d)

    #pick a random file from the list of files for each category and read them in
    #if mixed get even amounts of everything and use the label for that category
    for i,f in enumerate(files):
        #initialize the image
        full_img = cv2.imread(f,cv2.IMREAD_COLOR)
        img = cv2.resize(full_img,(constants.FULL_IMGSIZE,constants.FULL_IMGSIZE),interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        #extract features
        hogimg = getHOG(img)
        wt = extractWT(gray)

        #concatenate each channel and use it as the input
        img = np.concatenate((img,hogimg.reshape((hogimg.shape[0],hogimg.shape[1],1))),axis=-1)
        img = np.concatenate((img,wt.reshape((wt.shape[0],wt.shape[1],1))),axis=-1)

        if(catname == 'mixed'):
            #get pixel batch
            w, h, d = img.shape
            for j in range(n):
                low = int(constants.IMG_SIZE / 2)
                high = int(w - (constants.IMG_SIZE / 2) - 1)
                a = random.randint(low,high)
                b = random.randint(low,high)
                box_low1 = int(a - (constants.IMG_SIZE / 2))
                box_low2 = int(b - (constants.IMG_SIZE / 2))
                box_high1 = int(a + (constants.IMG_SIZE / 2))
                box_high2 = int(b + (constants.IMG_SIZE / 2))

                #get the box
                box = img[box_low1:box_high1,box_low2:box_high2,:]

                inputs.append(box)
                labels.append(categories[i])

    #if a category is chosen label is binary
    #get half of chosen category, half of others
        elif(catname == 'treematter' or catname == 'plywood' or catname == 'cardboard' or catname == 'trashbag' or catname == 'blackbag' or catname == 'bottles'):
            index = cats.index(catname)
            #even out the training between the label and the others
            if i == index:
                k = 1
            else:
                k = constants.CLASSES

            #get pixel instances
            w, h, d = img.shape
            for j in range(int(n/k)):
                #pick a random point
                low = int(constants.IMG_SIZE / 2)
                high = int(w - (constants.IMG_SIZE / 2) - 1)
                a = random.randint(low,high)
                b = random.randint(low,high)
                box_low1 = int(a - (constants.IMG_SIZE / 2))
                box_low2 = int(b - (constants.IMG_SIZE / 2))
                box_high1 = int(a + (constants.IMG_SIZE / 2))
                box_high2 = int(b + (constants.IMG_SIZE / 2))

                #get rgb
                box = img[box_low1:box_high1,box_low2:box_high2,:]

                #concatenate each channel and push it to the inputs
                inputs.append(box)

                #push the label of the input
                if i == index:
                    labels.append([1])
                else:
                    labels.append([0])

    #shuffle the input and labels in parralel
    c = list(zip(inputs,labels))
    random.shuffle(c)
    inputs,labels = zip(*c)
    inputs = np.array(inputs)[:n,:,:,:]
    labels = np.array(labels)[:n]

    #return as batch size to get normal distribution
    return inputs,labels

#extract wavelet transform of an image
def extractWT(image):
    #convert to float
    imArray =  np.float32(image)
    imArray /= 255;
    # compute coefficients
    coeffs=pywt.wavedec2(imArray, 'haar', level=1)

    #Process Coefficients
    coeffs_H=list(coeffs)
    coeffs_H[0] *= 0;

    # reconstruction
    imArray_H=pywt.waverec2(coeffs_H,'haar');
    imArray_H *= 255;
    imArray_H =  np.uint8(imArray_H)

    return imArray_H[:image.shape[0],:image.shape[1]]

#https://stackoverflow.com/questions/12729228/simple-efficient-bilinear-interpolation-of-images-in-numpy-and-python
#INPUT:
#im => image to process
#x => list of x indices to process in im
#y => list of y indices to process in im
#
#NOTES:
#   make sure to have the same size between x and y as they are the x,y coordinates to process on the image
def bilinear_interpolate(im, x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 3
    y0 = np.floor(y).astype(int)
    y1 = y0 + 3

    x0 = np.clip(x0, 0, im.shape[1] - 1);
    x1 = np.clip(x1, 0, im.shape[1] -1 );
    y0 = np.clip(y0, 0, im.shape[0]-1);
    y1 = np.clip(y1, 0, im.shape[0]-1);

    Ia = im[ y0, x0 ]
    Ib = im[ y1, x0 ]
    Ic = im[ y0, x1 ]
    Id = im[ y1, x1 ]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    return (wa*Ia + wb*Ib + wc*Ic + wd*Id).reshape((im.shape[0],im.shape[1]))

def getHOG(img):
    cellsize = (int(constants.IMG_SIZE / 2),int(constants.IMG_SIZE / 2))
    fd,hog_image = hog(img,orientations=8,pixels_per_cell=cellsize,cells_per_block=(1,1),block_norm='L2-Hys', visualize=True,multichannel=True)
    return hog_image
