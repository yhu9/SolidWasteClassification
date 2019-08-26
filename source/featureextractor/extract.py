#############################################################################################################
#Author: Masa Hu
#
#                               Trash Classification Project
#############################################################################################################
#NATIVE LIBRARY IMPORTS
import argparse
import time
import re
import os
import pickle
from multiprocessing import Process
from multiprocessing import Manager
import gc

#OPEN SOURCE IMPORTS
import numpy as np
import cv2

#CUSTOM IMPORTS
import gabor_threads_roi as gabor
import utils
#############################################################################################################
#ARGUMENT PARSER FOR FEATURES TO EXTRACT
parser = argparse.ArgumentParser()
parser.add_argument('--gabor',default=False,action='store_const',const=True)
parser.add_argument('--hog',default=False,action='store_const',const=True)
parser.add_argument('--rgb',default=False,action='store_const',const=True)
parser.add_argument('--hsv',default=False,action='store_const',const=True)
parser.add_argument('--size',default=False,action='store_const',const=True)
parser.add_argument('--file',type=str,required=True)
args = parser.parse_args()

#############################################################################################################
#FEATURE EXTRACTOR TO MULTIPROCESS
def extractFeatures(img,label,datadump):
    f1 = utils.getRGBHist(img)
    f2 = utils.getHSVHist(img)
    f3 = utils.getHOG(img)
    f4 = gabor.run_gabor(img,gabor.build_filters(16)).flatten()
    f5 = utils.getSize(img)
    datadump.append([np.hstack((f1,f2,f3,f4,f5)),label])

#HELPER FUNCTION TO GET LABEL BASED ON FILENAME
def getLabel(filename):
    group = re.findall("treematter|plywood|cardboard|bottles|trashbag|blackbag|mixed",filename)
    if(len(group) == 0): return -1
    elif(group[0] == 'treematter'): return 0
    elif(group[0] == 'plywood'): return 1
    elif(group[0] == 'cardboard'): return 2
    elif(group[0] == 'bottles'): return 3
    elif(group[0] == 'trashbag'): return 4
    elif(group[0] == 'blackbag'): return 5
    elif(group[0] == 'mixed'): return -1
    else: print('image belongs to no group'); return -1

#############################################################################################################
#############################################################################################################
if __name__ == '__main__':

    FILENAME = args.file
    img = cv2.imread(FILENAME)
    label = getLabel(FILENAME)

    #segment the image
    blobs, markers,labels = utils.extractBlobs(img)

    #multiprocess the feature extraction
    manager = Manager()
    datadump = manager.list()
    jobs = []
    max_process = 30
    for i,b in enumerate(blobs):
        p = Process(target=extractFeatures,args=(b,label,datadump))
        jobs.append(p)
        p.start()

        if i % max_process == max_process - 1:
            for j in jobs: j.join()
    for j in jobs: j.join()
    data = np.array([d[0] for d in datadump])
    label = np.array([d[1] for d in datadump])
    del datadump
    del jobs
    gc.collect()

    #PCA ANALYSIS ON DATA INSTANCES. POSSIBLY NOT NECESSARY?
    #data,pca = utils.getPCA(data)
    data = np.hstack((np.expand_dims(label,axis=1),data))
    print(np.amin(data))
    print(np.amax(data))

    #SAVE OUTPUT
    if not os.path.exists('out'):
        os.makedirs('out')
    np.save(os.path.join('out',FILENAME),data)
