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

def showSegmentSizeDistribution(image,markers):
    #remove markers given condition ange get unique markers again
    for k in size_dict.keys():
        if(size_dict[k] < mean):
            markers[markers == k] = 0
    uniqueMarkers = np.unique(markers)
    reduced_count = len(uniqueMarkers)

    #show the segmenting size selection process
    print("mean size: %s" % mean)
    print("segment counts: %s" % count)
    print("reduced counts: %s" % reduced_count)
    size_array.sort()
    size_hist = np.array(size_array)
    subset = size_hist[size_hist > mean]
    plt.figure(1)
    plt.subplot(211)
    plt.title('size distribution of segments')
    plt.plot(size_hist,'r--')

    plt.subplot(212)
    plt.title('size distribution after reduction')
    plt.plot(subset,'r--')
    plt.pause(0.1)
    cv2.waitKey(0)

#reduces rgb dimension into single dimension using bins
#just for my sanity sake i made this because I thought it might be very cool to see
def showImage3D(image,mode):
    #initialize variables
    height,width = image.shape[:2]
    bins = 8
    X = np.array(range(height))
    Y = np.array(range(width))
    x = []
    y = []
    color = []

    for i in X:
        for j in Y:
            pixel = image[i][j]
            if(isinstance(pixel,np.uint8)):
                val = pixel
                x.append(i)
                y.append(j)
                color.append(val)
            else:
                val = 0
                if(mode == 'b'):
                    val = pixel[0]
                elif(mode == 'g'):
                    val = pixel[1]
                elif(mode == 'r'):
                    val = pixel[2]
                elif(mode == 'bin'):
                    b = pixel[0]
                    g = pixel[1]
                    r = pixel[2]

                    bbin = int(float(b) / float(256) * float(bins))
                    gbin = int(float(g) / float(256) * float(bins))
                    rbin = int(float(r) / float(256) * float(bins))
                    for a,bval in enumerate([bbin,gbin,rbin]):
                        val += bval * pow(bins,a)
                else:
                    val = 1

                x.append(i)
                y.append(j)
                color.append(val)


    #create the x,y,z axis of length x * y
    xcoor = np.array(x)
    ycoor = np.array(y)
    Z = np.array(color)
    #For use with the contour map but I get an error "out of memory" when using meshgrid

    #create the figure
    fig = plt.figure()
    plt.title('3-d visualization of 2d image using color bins')

    #show the 3D scatter plot.
    #if i can ever figure out how to get the countour map to work I will do so.
    ax = plt.axes(projection = '3d')
    ax.view_init(-90, 180)
    ax.scatter(xcoor,ycoor,Z,c=Z,cmap='viridis')
    #ax.contour3D(X,Y,Z,50,cmap='viridis')

    plt.show()

#reduces rgb dimension into single dimension using bins
#just for my sanity sake i made this because I thought it might be very cool to see
'''
inputs:
    1. image to be converted to collection of 3d points
output:
    1. numpy 2d array with collection of 3d points
'''
def cvtImage3DPoints(image,method):
    #initialize variables
    if(len(image.shape) != 3):
        print('invalid image depth! Cannot be greyscaled image')

    #get image shape
    height,width,depth = image.shape
    bins = 8
    X = np.array(range(height))
    Y = np.array(range(width))
    points = np.copy(image)

    #define broadcasting function to extract bgr information
    def foo(pixel,method='binning'):
        val = 0
        b = pixel[0]
        g = pixel[1]
        r = pixel[2]
        print(pixel)
        if(method == 'binning'):

            bbin = int(float(b) / float(256) * float(bins))
            gbin = int(float(g) / float(256) * float(bins))
            rbin = int(float(r) / float(256) * float(bins))
            for a,bval in enumerate([bbin,gbin,rbin]):
                val += bval * pow(bins,a)

        #https://stackoverflow.com/questions/687261/converting-rgb-to-grayscale-intensity
        #uses human perception of color intensities?
        elif(method == 'original'):
            val = 0.2989 * r + 0.5870 * g + 0.1140 * b

        return val

    #broadcast the points and convert 2d image into n feature array of points and return it as a
    #(1,h * w, d) numpy array
    for i in X:
        points[:,i,1] = i
    for j in Y:
        points[:,j,0] = j

    np.vectorize(image,foo)
    #print(points)
    #points[:,:,2] = foo(image,method=method)
    #print(image)
    #print(foo(image))

    return np.reshape(points,(1,height * width,depth))

#reduces rgb dimension into single dimension using bins
#just for my sanity sake i made this because I thought it might be very cool to see
#http://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
'''
inputs:
    1. image to be segmented then shown
output:
    1. collection of points and the labels for each point
'''
def dbscan(image,binning=False):

    if(binning):
        points = cvtImage3DPoints(image,'binning')
    else:
        points = cvtImage3DPoints(image,'original')
    #http://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
    height,width,depth = image.shape
    db = DBSCAN(
            eps=30,
            min_samples=200,
            metric='euclidean',
            metric_params=None,
            algorithm='auto',
            leaf_size=30,
            p=None,
            n_jobs=-1
            )

    db.fit(points)
    labels = db.labels_
    cluster_count = len(np.unique(labels))

    print("Number of estimated clusters: ", cluster_count)

    label_image = np.reshape(labels,(height,width))

    return image,label_image

#reduces rgb dimension into single dimension using bins
#just for my sanity sake i made this because I thought it might be very cool to see
#http://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html
'''
inputs:
    1. image to be segmented then shown
output:
    1. collection of points and the labels for each point
'''
def meanshift(image,binning=False):

    if(binning):
        points = cvtImage3DPoints(image,'binning')
    else:
        points = cvtImage3DPoints(image,'original')
    #http://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html
    height,width,depth = image.shape
    bw = estimate_bandwidth(points,quantile=0.2,n_samples=((height * width) / 2))
    ms = MeanShift(
            bandwidth=100,
            seeds=None,
            bin_seeding=True,
            min_bin_freq=100,
            cluster_all=True,
            n_jobs=-1
            )

    ms.fit(points)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    cluster_count = len(np.unique(labels))

    print("Number of estimated clusters: ", cluster_count)

    label_image = np.reshape(labels,(height,width))

    return image,label_image

#According to http://stackoverflow.com/questions/17063042/why-do-we-convert-from-rgb-to-hsv/17063317
#HSV is better for object recognition compared to BGR
# H max = 170
# S max = 255
# V max = 255
def extractHSVHist(imageIn,SHOW):

    color = ('h','s','v')
    hist = []
    for i,col in enumerate(color):
        if col == 'h':
            series = cv2.calcHist([imageIn],[i],None,[170],[0,170])
        else:
            series = cv2.calcHist([imageIn],[i],None,[256],[0,256])

        hist.append(np.ravel(series)[1:])

    #show the results
    if(SHOW):
        cv2.namedWindow('Processing Segment',cv2.WINDOW_NORMAL)
        cv2.imshow('Processing Segment',imageIn)   #
        plt.plot(hist)
        plt.show()

    #lop off black and white
    return np.concatenate(np.array(hist))

def extractColorBinHist(imageIn,SHOW):
    bins = 8
    colors = np.zeros((bins,bins,bins))
    height,width = imageIn.shape[:2]

    #create the histogram of bins^3 colors
    for i in range(height):
        for j in range(width):
            pixel = imageIn[i][j]
            if(isinstance(pixel,np.uint8)):
                b = pixel
                g = pixel
                r = pixel
                bbin = int(float(b) / float(256/bins))
                gbin = int(float(g) / float(256/bins))
                rbin = int(float(r) / float(256/bins))

                colors[bbin][gbin][rbin] += 1
            else:
                b = pixel[0]
                g = pixel[1]
                r = pixel[2]

                bbin = int(float(b) / float(256/bins))
                gbin = int(float(g) / float(256/bins))
                rbin = int(float(r) / float(256/bins))

                colors[bbin][gbin][rbin] += 1

    #flatten the 3-d feature fector into 1-d
    hist = colors.flatten()

    #show the results
    if(SHOW):
        cv2.namedWindow('Processing Segment',cv2.WINDOW_NORMAL)
        cv2.imshow('Processing Segment',imageIn)   #
        plt.plot(hist)
        plt.show()

    #lop off black and white
    return hist[1:-1]

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
    for i,col in enumerate(color):
        series = cv2.calcHist([imageIn],[i],None,[256],[0,256])
        hist.append(np.ravel(series)[1:])

    #show the results
    if(SHOW):
        cv2.namedWindow('Processing Segment',cv2.WINDOW_NORMAL)
        cv2.imshow('Processing Segment',imageIn)   #
        plt.plot(hist)
        plt.show()

    #lop off black and white
    return np.concatenate(np.array(hist))


#get the blob size from the blob
'''
Inputs:
    1. image
Outputs:
    1. blob size
'''
def extractBlobSize(image):
    blob_size = np.count_nonzero(image != [0,0,0])

    return np.array([blob_size])

#extract the edge distribution from the image segment
def extractHOG(imageIn, SHOW):
    #necessary for seeing the plots in sequence with one click of a key

    h,w = imageIn.shape[:2]
    new_w = (int(int(w) / int(16)) + 1 ) * 16
    new_h = (int(int(h) / int(16)) + 1 ) * 16

    #resize the image to 64 x 128
    resized = cv2.resize(imageIn,(new_w, new_h), interpolation = cv2.INTER_CUBIC)

    #HOG DESCRIPTOR INITILIZATION
    #https://stackoverflow.com/questions/28390614/opencv-hogdescripter-python
    #https://docs.opencv.org/2.4/modules/gpu/doc/object_detection.html
    #https://www.learnopencv.com/histogram-of-oriented-gradients/
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
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                                    histogramNormType,L2HysThreshold,gammaCorrection,nlevels)

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
    pca = PCA(copy=True, iterated_power='auto', n_components=0.95, random_state=None,
              svd_solver='auto', tol=0.0, whiten=False)

    #apply pca on the feature vector
    new_vector = pca.fit_transform(featurevector)

    #save pca details to text file
    np.set_printoptions(threshold=np.inf)
    if SAVEFLAG:
        with open(fnameout,'w') as fout:
            fout.write('------------------------------------------------------------------\n')
            fout.write('------------------------------------------------------------------\n')
            fout.write(" EXPLAINED VARIANCE RATIO\n\n\n")
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
            fout.write("PCA FEATURE COUNT FOUND\n\n\n")
            fout.write('' + str(len(pca.singular_values_)) + '\n')
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
def writeFeatures(features, fnameout='output.txt', label=None):

    if len(features) == 0 or type(features) != type(np.array([])):
        print ("features type: %s" % type(features))
        print ("expected type: %s" % type(np.array([])))
        print ("length features: %i" % len(features))
        print ("error with the input to the extractionModule.writeFeatures()")
        return False
    else:
        if len(features.shape) == 1:
            with open(fnameout,'w') as fout:
                for i,val in enumerate(features):
                    fout.write(str(val))
                    if i < len(features) - 1:
                        fout.write(",")
                if label is not None:
                    fout.write("  #%s#" % label)
                fout.write('\n')
        elif len(features.shape) == 2:
            with open(fnameout,'w') as fout:
                for i,instance in enumerate(features):
                    for j,val in enumerate(instance):
                        fout.write(str(val))
                        if j < len(features) - 1:
                            fout.write(",")
                    if label is not None:
                        fout.write("  #%s#" % label[i])
                    fout.write('\n')
        else:
            print('unhandled data dimension!')
            print("please convert data into 1D or 2D array")

    return True

#Display a histogram
def displayHistogram(hist,normalize=False):
    plt.figure()
    plt.plot(hist)
    plt.show()

#normalize values to max of the set
def normalize(instances):
    norm_instances = instances.astype(np.float) / np.amax(instances)
    return norm_instances

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
