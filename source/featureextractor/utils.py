#NATIVE LIBRARY IMPORTS
import random

#OPEN SOURCE IMPORTS
import cv2
import numpy as np
import pymeanshift as pms
from sklearn.decomposition import PCA

########################################################################
#Canny image
#http://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?highlight=canny
########################################################################
#Parameters:
#    image - single-channel 8-bit input image.
#    edges - output edge map; it has the same size and type as image .
#    threshold1 - first threshold for the hysteresis procedure.
#    threshold2 - second threshold for the hysteresis procedure.
#    apertureSize - aperture size for the Sobel() operator.
#    L2gradient - a flag, indicating whether a more accurate L_2 norm =\sqrt{(dI/dx)^2 + (dI/dy)^2} should be used to calculate the image gradient magnitude ( L2gradient=true ), or whether the default L_1 norm =|dI/dx|+|dI/dy| is enough ( L2gradient=false ).
def getSegments(original, SHOW=False,SPATIAL_RADIUS=5,RANGE_RADIUS=5,MIN_DENSITY=10000):
    ##############################################################################################################
    #gaussian Blur
    #blur = cv2.GaussianBlur(gray_img,(5,5),0)

    #mean shift segmentation on bgr image
    #https://github.com/fjean/pymeanshift
    #http://ieeexplore.ieee.org/document/1000236/
    segmented_image,labels_image,number_regions = pms.segment(
            original,
            spatial_radius=SPATIAL_RADIUS,
            range_radius=RANGE_RADIUS,
            min_density=MIN_DENSITY,
            speedup_level=2)
    print("Number of Regions Found: %s" % number_regions)
    unique_labels = np.unique(labels_image)
    blank = original - original
    for label in unique_labels:
        b = random.randint(0,255)
        g = random.randint(0,255)
        r = random.randint(0,255)
        blank[ labels_image == label] = [b,g,r]

    if SHOW == "save":
        cv2.imwrite("saved_segmentation.png",blank)

    ################################################################################
    ################################################################################
    ################################################################################
    ################################################################################
    ################################################################################

    return segmented_image, labels_image

'''
INPUTS:
    1. image
OUTPUTS:
    1. 3d numpy array with blobs as instances as 2d numpy arrays
    2. 1d numpy array as labels
'''
def extractBlobs(img, mode='hsv'):
    blobs = []
    if mode == 'hsv':
        print('HSV SEGMENTATION')
        hsvimg = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        seg_img,markers = getSegments(hsvimg)
    elif mode == 'bgr':
        print('BGR SEGMENTATION')
        seg_img,markers = getSegments(img,SPATIAL_RADIUS=1,RANGE_RADIUS=1)

    labels = np.unique(markers)
    canvas = img.copy()
    for uq_mark in labels[1:]:
        #get the segment and append it to inputs
        region = img.copy()
        region[markers != uq_mark] = [0,0,0]

        #opencv bounding rect only works on single channel images...
        blank = img.copy()
        blank = blank - blank
        blank[markers == uq_mark] = [255,255,255]
        grey = cv2.cvtColor(blank,cv2.COLOR_BGR2GRAY)
        x,y,w,h = cv2.boundingRect(grey)
        cropped = region[y:y+h,x:x+w]
        cropped = np.uint8(cropped)
        blob = cv2.resize(cropped,(64,64),interpolation=cv2.INTER_LINEAR)
        blobs.append(blob)

        b = random.randint(0,255)
        g = random.randint(0,255)
        r = random.randint(0,255)
        canvas[markers == uq_mark] = [b,g,r]

    return blobs, markers, labels

#GET PCA FEATURES OVER SEVERAL INSTANCES OF 1-D FEATURE VECTORS
def getPCA(vec):
    pca = PCA(n_components=0.99)
    newfeatures = pca.fit_transform(vec)
    return newfeatures,pca

###################################################################################################
#FEATURE EXTRACTION UTILITY FUNCTIONS
###################################################################################################
#GET RGB FEATURE VECTOR
def getRGBHist(imageIn):

    h,w,d = imageIn.shape
    color = ('b','g','r')
    hist = []
    zeropix = np.count_nonzero(np.all(imageIn == [0,0,0],axis=2))
    for i,col in enumerate(color):
        series = cv2.calcHist([imageIn],[i],None,[256],[0,256])
        series[0] = series[0] - zeropix
        hist.append(np.ravel(series))

    return np.concatenate(np.array(hist)) / (h*w)

#GET HSV FEATURE VECTOR
def getHSVHist(imageIn):
    h,w,d = imageIn.shape
    color = ('h','s','v')
    hist = []
    zeropix = np.count_nonzero(np.all(imageIn == [0,0,0],axis=2))
    for i,col in enumerate(color):
        if col == 'h':
            series = cv2.calcHist([imageIn],[i],None,[170],[0,170])
        else:
            series = cv2.calcHist([imageIn],[i],None,[256],[0,256])

        series[0] -= zeropix
        hist.append(np.ravel(series))

    return np.concatenate(np.array(hist)) / (h*w)

#GET HOG FEATURES
#https://stackoverflow.com/questions/28390614/opencv-hogdescripter-python
#https://docs.opencv.org/2.4/modules/gpu/doc/object_detection.html
#https://www.learnopencv.com/histogram-of-oriented-gradients/
def getHOG(imageIn):
    h,w,d = imageIn.shape
    new_w = (int(int(w) / int(16)) + 1 ) * 16
    new_h = (int(int(h) / int(16)) + 1 ) * 16

    #resize the image to 64 x 128
    resized = cv2.resize(imageIn,(new_w, new_h), interpolation = cv2.INTER_CUBIC)

    #HOG DESCRIPTOR INITILIZATION
    winSize = (new_w,new_h)                               #
    blockSize = (16,16)                             #only 16x16 block size supported for normalization
    blockStride = (8,8)                             #only 8x8 block stride supported
    cellSize = (8,8)                                #individual cell size should be 1/4 of the block size
    nbins = 9                                       #only 9 supported over 0 - 180 degrees
    derivAperture = 1                               #
    winSigma = 4.                                   #
    histogramNormType = 0                           #
    L2HysThreshold = 2.0000000000000001e-01         #L2 normalization exponent ex: sqrt(x^L2 + y^L2 + z^L2)
    gammaCorrection = 0                             #
    nlevels = 64                                    #
    cvhog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                                    histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
    hist = cvhog.compute(resized)

    #create the feature vector
    feature = []
    for i in range(nbins):
        feature.append(0)
    for i in range(len(hist)):
        feature[i % (nbins)] += hist[i]
    feature_hist = np.array(feature)

    #show the results of the HOG distribution for the section. SANITY CHECK
    #if(SHOW):
    #    cv2.namedWindow('Processing Segment',cv2.WINDOW_NORMAL)
    #    cv2.imshow('Processing Segment',imageIn)   #
    #    plt.plot(feature_hist)
    #    plt.draw()
    #    plt.show()

    return feature_hist.ravel() / np.sum(feature_hist)

#GETS THE NUMBER OF PIXELS THAT ARE NON-BLACK
def getSize(image):
    h,w,d = image.shape
    blob_size = np.count_nonzero(np.all(image != [0,0,0],axis=2))

    return np.array([blob_size]) / (h*w)
