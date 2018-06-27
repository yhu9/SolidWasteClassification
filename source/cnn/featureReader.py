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
    #if mixed get even amount of everything and use the label for that category
    if(catname == 'mixed'):
        for i,f in enumerate(files):
            full_img = cv2.imread(f,cv2.IMREAD_COLOR)
            img = cv2.resize(full_img,(constants.FULL_IMGSIZE,constants.FULL_IMGSIZE),interpolation=cv2.INTER_CUBIC)

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

                box = img[box_low1:box_high1,box_low2:box_high2]
                inputs.append(box)
                labels.append(categories[i])

    #if a category is chosen label is binary
    #get half of chosen category, half of others
    elif(catname == 'treematter' or catname == 'plywood' or catname == 'cardboard' or catname == 'trashbag' or catname == 'blackbag' or catname == 'bottles'):
        index = cats.index(catname)
        for i,f in enumerate(files):
            img = cv2.imread(f,cv2.IMREAD_COLOR)
            w, h, d = img.shape
            if i == index:
                k = 1
            else:
                k = constants.CLASSES
            for j in range(int(n/k)):
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
                if i == index:
                    labels.append([1])
                else:
                    labels.append([0])

    #shuffle the input and labels in parralel
    c = list(zip(inputs,labels))
    random.shuffle(c)
    inputs,labels = zip(*c)

    #return as batch size to get normal distribution
    return np.array(inputs)[:n],np.array(labels)[:n]

