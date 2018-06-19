#############################################################################################################
#Masa Hu
#
#                               Trash Classification Project
#############################################################################################################
#Library imports
import cv2
import numpy as np
import math
import gabor_threads_roi as gabor
from matplotlib import pyplot as plt
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.cluster import MeanShift
from sklearn.cluster import estimate_bandwidth
from sklearn.cluster import DBSCAN
from matplotlib import style

style.use("ggplot")

#############################################################################################################
#                               Description of Module
#
#The output of the exported extract features function is:
#   array([np.array,np.array,np.array], ...]    features
#
#brief description:
#
#this module takes a source image with marked regions and extracts HSV color histograms
#as features for each region defined in markers. The HSV color value [0,0,0] gets dropped
#due to constraints on the opencv calcHist() function which must take a rectangular image
#as the input parameter. Since, marked regions are not rectangular, a copy of the original image
#is used with a particular marking to make an image that is all black except for the specified
#region. This image is then used to extract the histogram distribution of that region. This process
#is repeated until all regions are stored in features.
#
#Setting the show flag allows the user to specify how slowly they would like to see the histogram
#distribution extraction for each region.
#############################################################################################################
#############################################################################################################

#The function extractFeatures() takes in the inputs:
#   Mat         image
#   np.array    markers
#   bool        SHOW
#
#The output of the exported extract features function is a 1-d np array
#
#According to http://stackoverflow.com/questions/17063042/why-do-we-convert-from-rgb-to-hsv/17063317
#HSV is better for object recognition compared to BGR
# H max = 170
# S max = 255
# V max = 255
def extractColorHist(imageIn,SHOW):

    color = ('b','g','r')
    hist = []
    zeropix = np.count_nonzero(np.all(imageIn == [0,0,0],axis=2))
    for i,col in enumerate(color):
        series = cv2.calcHist([imageIn],[i],None,[256],[0,255])
        #series[0] = series[0] - zeropix
        hist.append(np.ravel(series))

    #show the results
    if(SHOW):
        cv2.namedWindow('Processing Segment',cv2.WINDOW_NORMAL)
        cv2.imshow('Processing Segment',imageIn)   #
        plt.plot(hist)
        plt.show()

    #lop off black and white
    return np.concatenate(np.array(hist))

#According to http://stackoverflow.com/questions/17063042/why-do-we-convert-from-rgb-to-hsv/17063317
#HSV is better for object recognition compared to BGR
# H max = 170
# S max = 255
# V max = 255
def extractHSVHist(imageIn,SHOW):

    color = ('h','s','v')
    hist = []
    zeropix = np.count_nonzero(np.all(imageIn == [0,0,0],axis=2))
    for i,col in enumerate(color):
        if col == 'h':
            series = cv2.calcHist([imageIn],[i],None,[170],[0,170])
        else:
            series = cv2.calcHist([imageIn],[i],None,[256],[0,256])

        series[0] -= zeropix
        hist.append(np.ravel(series)[1:])

    #show the results
    if(SHOW):
        cv2.namedWindow('Processing Segment',cv2.WINDOW_NORMAL)
        cv2.imshow('Processing Segment',imageIn)   #
        plt.plot(hist)
        plt.show()

    #lop off black and white
    return np.concatenate(np.array(hist))

#extract the edge distribution from the image segment
def extractHOG(imageIn, SHOW):
    #necessary for seeing the plots in sequence with one click of a key

    h,w,d = imageIn.shape
    new_w = int(int((int(w) / 16.0) + 1 ) * 16)
    new_h = int(int((int(h) / 16.0) + 1 ) * 16)

    #resize the image to 64 x 128
    resized = cv2.resize(imageIn,(int(new_w), int(new_h)), interpolation = cv2.INTER_CUBIC)

    #HOG DESCRIPTOR INITILIZATION
    #https://stackoverflow.com/questions/28390614/opencv-hogdescripter-python
    #https://docs.opencv.org/2.4/modules/gpu/doc/object_detection.html
    #https://www.learnopencv.com/histogram-of-oriented-gradients/
    #sometimes you get this error when trying to run it
    #https://stackoverflow.com/questions/42448628/opencv-version-3-hogdescriptor-takes-at-most-1-argument-5-given
    winSize = (int(new_w),int(new_h))                               #
    blockSize = (int(16),int(16))                             #only 16x16 block size supported for normalization
    blockStride = (int(8),int(8))                             #only 8x8 block stride supported
    cellSize = (8,8)                                #individual cell size should be 1/4 of the block size
    nbins = 9                                       #only 9 supported over 0 - 180 degrees
    derivAperture = 1                               #
    winSigma = 4                                   #
    histogramNormType = 0                           #
    L2HysThreshold = 2.0000000000000001e-01
    gammaCorrection = 0                             #
    nlevels = 64                                    #
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels)

    hist = hog.compute(resized)

    #create the feature vector
    feature = []
    for i in range(nbins):
        feature.append(0)
    for i in range(len(hist)):
        feature[i % (nbins)] += hist[i]
    feature_hist = np.array(feature)
    feature_hist = feature / np.amax(feature)

    #show the results of the HOG distribution for the section
    if(SHOW):
        cv2.namedWindow('Processing Segment',cv2.WINDOW_NORMAL)
        cv2.imshow('Processing Segment',imageIn)   #
        plt.plot(feature_hist)
        plt.draw()
        plt.show()

    return feature_hist


#get the blob size from the blob
'''
Inputs:
    1. image
Outputs:
    1. blob size
'''
def extractBlobSize(image):
    blob_size = np.count_nonzero(np.all(image != [0,0,0]))

    return np.array([blob_size])


#applies pca analysis to feature vector with n instances and vector length i
'''
Inputs:
   1. 1D feature vector
Outputs:
    2. new 1D feature vector
'''
def ldaAnalysis(featurevector,SAVEFLAG=True, fnameout='pca_details.txt'):
    instance_count,feature_count = featurevector.shape

    #apply pca on the feature vector
    new_vector = pca.fit_transform(featurevector)

    #save pca details to text file
    np.set_printoptions(threshold=np.inf)
    if SAVEFLAG:
        with open(fnameout,'w') as fout:
            fout.write('------------------------------------------------------------------\n')
            fout.write('------------------------------------------------------------------\n')
            fout.write("PCA EXPLAINED VARIANCE RATIO\n\n\n")
            fout.write('' + str(pca.explained_variance_ratio_) + '\n')
            fout.write('------------------------------------------------------------------\n')
            fout.write('------------------------------------------------------------------\n')
            fout.write("PCA SINGULAR VALUES\n\n\n")
            fout.write('' + str(pca.singular_values_) + '\n')
            fout.write('------------------------------------------------------------------\n')
            fout.write('------------------------------------------------------------------\n')
            fout.write("PCA STACKED COVARIANCE VALUES\n\n\n")
            fout.write('' + str(np.sum(pca.get_covariance(),axis=1)) + '\n')
            fout.write('------------------------------------------------------------------\n')
            fout.write('------------------------------------------------------------------\n')
            fout.write("PCA COMPONENTS\n\n\n")
            fout.write('' + str(pca.components_) + '\n')
            fout.write('------------------------------------------------------------------\n')
            fout.write('------------------------------------------------------------------\n')

    #PRINT OUT PCA VALUES to console
    print('-------------------------------------------------------------------------------------')
    print('-------------------------------------------------------------------------------------')
    print('-------------------------------------------------------------------------------------')
    print("PCA EXPLAINED VARIANCE RATIO\n\n\n")
    print('' + str(pca.explained_variance_ratio_) + '\n')
    print('-------------------------------------------------------------------------------------')
    print('-------------------------------------------------------------------------------------')
    print('-------------------------------------------------------------------------------------')
    print("PCA SINGULAR VALUES\n\n\n")
    print('' + str(pca.singular_values_) + '\n')
    print('-------------------------------------------------------------------------------------')
    print('-------------------------------------------------------------------------------------')
    print('-------------------------------------------------------------------------------------')
    print("PCA STACKED COVARIANCE VALUES\n\n\n")
    print('' + str(np.sum(pca.get_covariance(),axis=1)) + '\n')
    print('-------------------------------------------------------------------------------------')
    print('-------------------------------------------------------------------------------------')
    print('-------------------------------------------------------------------------------------')
    print("PCA FEATURE COUNT FOUND\n\n\n")
    print('' + str(len(pca.singular_values_)) + '\n')
    print('------------------------------------------------------------------\n')
    print('------------------------------------------------------------------\n')

    return new_vector

#applies pca analysis to feature vector with n instances and vector length i
'''
Inputs:
   1. 1D feature vector
Outputs:
    2. new 1D feature vector
'''
def pcaAnalysis(featurevector,SAVEFLAG=True, fnameout='pca_details.txt'):
    instance_count,feature_count = featurevector.shape

    #http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    #all default values except for n_components
    pca = PCA(copy=True, iterated_power='auto', n_components=0.99, random_state=None,
              svd_solver='auto', tol=0.0, whiten=False)

    #apply pca on the feature vector
    new_vector = pca.fit_transform(featurevector)

    #save pca details to text file
    np.set_printoptions(threshold=np.inf)
    if SAVEFLAG:
        with open(fnameout,'w') as fout:
            fout.write('------------------------------------------------------------------\n')
            fout.write('------------------------------------------------------------------\n')
            fout.write("PCA EXPLAINED VARIANCE RATIO\n\n\n")
            fout.write('' + str(pca.explained_variance_ratio_) + '\n')
            fout.write('------------------------------------------------------------------\n')
            fout.write('------------------------------------------------------------------\n')
            fout.write("PCA SINGULAR VALUES\n\n\n")
            fout.write('' + str(pca.singular_values_) + '\n')
            fout.write('------------------------------------------------------------------\n')
            fout.write('------------------------------------------------------------------\n')
            fout.write("PCA STACKED COVARIANCE VALUES\n\n\n")
            fout.write('' + str(np.sum(pca.get_covariance(),axis=1)) + '\n')
            fout.write('------------------------------------------------------------------\n')
            fout.write('------------------------------------------------------------------\n')
            fout.write("PCA COMPONENTS\n\n\n")
            fout.write('' + str(pca.components_) + '\n')
            fout.write('------------------------------------------------------------------\n')
            fout.write('------------------------------------------------------------------\n')

    #PRINT OUT PCA VALUES to console
    print('-------------------------------------------------------------------------------------')
    print('-------------------------------------------------------------------------------------')
    print('-------------------------------------------------------------------------------------')
    print("PCA EXPLAINED VARIANCE RATIO\n\n\n")
    print('' + str(pca.explained_variance_ratio_) + '\n')
    print('-------------------------------------------------------------------------------------')
    print('-------------------------------------------------------------------------------------')
    print('-------------------------------------------------------------------------------------')
    print("PCA SINGULAR VALUES\n\n\n")
    print('' + str(pca.singular_values_) + '\n')
    print('-------------------------------------------------------------------------------------')
    print('-------------------------------------------------------------------------------------')
    print('-------------------------------------------------------------------------------------')
    print("PCA STACKED COVARIANCE VALUES\n\n\n")
    print('' + str(np.sum(pca.get_covariance(),axis=1)) + '\n')
    print('-------------------------------------------------------------------------------------')
    print('-------------------------------------------------------------------------------------')
    print('-------------------------------------------------------------------------------------')
    print("PCA FEATURE COUNT FOUND\n\n\n")
    print('' + str(len(pca.singular_values_)) + '\n')
    print('------------------------------------------------------------------\n')
    print('------------------------------------------------------------------\n')

    return new_vector

#Writes the features out to a file called extraction_out.txt in the working directory by default
'''
INPUT:
    1. features to write out
    2. (option) file name to write the features to
OUTPUT:
    1. True
'''
def writeFeatures(features, fnameout='output', label=None):

    if len(features) == 0 or type(features) != type(np.array([])):
        print ("features type: %s" % type(features))
        print ("expected type: %s" % type(np.array([])))
        print ("length features: %i" % len(features))
        print ("error with the input to the extractionModule.writeFeatures()")
        return False
    else:
        if label is not None:

            tmp = np.hstack((features,label))
            np.save(fnameout,tmp)

    return True

#Display a histogram
def displayHistogram(hist,normalize=False):
    plt.figure()
    plt.plot(hist)
    plt.show()

#normalize values to max of the set
def normalize(instances):
    norm_instances = instances.astype(np.float) / np.amax(instances)
    return np.nan_to_num(norm_instances)

#takes a single image and extracts all features depending on flag constants
#based on user input
'''
INPUTS:
    1. segment of type numpy array
    2. (optional) hogflag  of type bool
    3. (optional) gaborflag of type bool
    4. (optional) colorflag of type bool
OUTPUTS:
    1. feature vector
'''
def evaluateSegment(segment,hogflag=False,gaborflag=False,colorflag=False,sizeflag=False,hsvflag=False):
    #extract features for each image depending on the flag constants
    features = []

    if sizeflag:
        features.append(evaluate(segment,'size'))
    if hogflag:
        features.append(evaluate(segment,'hog'))
    if gaborflag:
        features.append(evaluate(segment,'gabor'))
    if colorflag:
        features.append(evaluate(segment,'color'))
    if hsvflag:
        features.append(evaluate(segment,'hsv'))

    #create the full feature vector for the given instance image and push to instances
    #and also push the file name as the label for the instance
    full_vector = np.array([])
    for i in range(len(features)):
        full_vector = np.hstack((full_vector,features[i]))

    return full_vector

#EVALUATE AN IMAGE GIVEN THE MODE
def evaluate(original,mode,SHOWFLAG=False):
    #check if the image was read in correctly
    if original is None:
        print('invalid image! Could not open image')

    #if mode is size we have to normalize this later across all instances
    if mode == 'size':
        combined_filename = sys.argv[1]

        # Generate and save blob size for this blob we assume black as background
        size = extractBlobSize(original)
        #print('--------------SIZE---------------')
        return size

    #if mode is hog, show hog feature vector of image
    elif mode == 'hog':
        hist = extractHOG(original,False)
        featurevector = hist.flatten()
        norm = normalize(featurevector)
        #print('-------------HOG----------------')
        #print(norm)
        if SHOWFLAG:
            displayHistogram(norm)
        return norm

    #if mode is color, show color histogram of image
    elif mode == 'color':
        hist = extractColorHist(original,False)
        norm = normalize(hist)
        #print('-------------Color----------------')
        #print(norm)
        if SHOWFLAG:
            displayHistogram(norm)
        return norm

    #if mode is gabor, extract gabor feature from image using several orientations
    elif mode == 'gabor':
        orientations = 16
        filters = gabor.build_filters(orientations)
        combined_filename = sys.argv[1]

        # Generate and save ALL hogs for this image
        result = gabor.run_gabor(original, filters, combined_filename, orientations, mode='training')
        featurevector = result.flatten()[1:]
        norm = normalize(featurevector)
        #print('--------------Gabor---------------')
        #print(norm)
        if SHOWFLAG:
            displayHistogram(norm,'r--')
        return norm

    elif mode == 'hsv':
        hsvimg = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
        hist = extractHSVHist(hsvimg,False)
        norm = normalize(hist)
        return norm
