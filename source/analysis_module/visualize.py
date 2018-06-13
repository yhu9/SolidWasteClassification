import sys
import segmentModule
import extractionModule as analyze
import numpy as np
import cv2
import time
import re
import gabor_threads_roi as gabor
import os

#GLOBAL FLAG VARIABLES
#flags that are handled
showflag = 'show' in sys.argv
hogflag = 'hog' in sys.argv
binflag = 'bin' in sys.argv
sizeflag = 'size' in sys.argv
colorflag = 'color' in sys.argv
gaborflag = 'gabor' in sys.argv
hsvflag = 'hsv' in sys.argv
msflag = 'meanshift' in sys.argv
msbinflag = 'meanshiftbin' in sys.argv
fjmsflag = 'fjmeanshift' in sys.argv
dbscanflag = 'dbscan' in sys.argv
pcaflag = 'pca' in sys.argv
ldaflag = 'lda' in sys.argv

#Check system argument length and mode
#if mode is bin do 3d color binning
start_time = time.time()

#display different modes
def display(original,labels=None,SHOWFLAG=True):
    if original is None:
        print('invalid image! Could not open: %s' % full_path)

    #if mode is meanshift, apply meanshift
    elif msflag:
        image, labels = analyze.meanshift(original)
        print(labels)
        if SHOWFLAG:
            segmentModule.showSegments(image,labels)

    #if mode is meanshiftbin, convert 2d image to 3d using bin method and apply meanshift
    elif msbinflag:
        image,labels = analyze.meanshift(original,binning=True)
        print(labels)
        if SHOWFLAG:
            segmentModule.showSegments(image,labels)

    #if mode is fjmeanshift, do fjmeanshift
    elif fjmsflag:
        if SHOWFLAG:
            image,labels = segmentModule.getSegments(original,True)
        else:
            image,labels = segmentModule.getSegments(original,False)
        print(labels)

    #if mode is meanshiftbin, convert 2d image to 3d using bin method and apply meanshift
    elif dbscanflag:
        image,labels = analyze.dbscan(original,binning=True)
        print(labels)
        if SHOWFLAG:
            segmentModule.showSegments(image,labels)

    #if mode is size
    elif sizeflag:
        combined_filename = sys.argv[1]

        # Generate and save blob size for this blob we assume black as background
        size = analyze.extractBlobSize(original)
        print('--------------SIZE---------------')
        if SHOWFLAG:
            print(size)
        return size

    #if mode is hog, show hog feature vector of image
    elif hogflag:
        hist = analyze.extractHOG(original,False)
        featurevector = hist.flatten()
        norm = analyze.normalize(featurevector)
        print('-------------HOG----------------')
        if SHOWFLAG:
            analyze.displayHistogram(featurevector)
        return norm

    #if mode is gabor, extract gabor feature from image using several orientations
    elif gaborflag:
        orientations = 16
        filters = gabor.build_filters(orientations)
        combined_filename = sys.argv[1]

        # Generate and save ALL hogs for this image
        result = gabor.run_gabor(original, filters, combined_filename, orientations, mode='training')
        featurevector = result.flatten()[1:]
        norm = analyze.normalize(featurevector)
        print('--------------Gabor---------------')
        if SHOWFLAG:
            analyze.displayHistogram(featurevector,'r--')
        return norm

    #if mode is color, show color histogram of image
    elif colorflag:
        hist = analyze.extractColorHist(original,False)
        print('-------------Color----------------')
        if SHOWFLAG:
            analyze.displayHistogram(hist)
        return hist

    elif binflag:
        hist = analyze.extractbinHist(original,False)
        norm = analyze.normalize(hist)
        if SHOWFLAG:
            analyze.displayHistogram(norm)
        return norm

    elif hsvflag:
        hsvimg = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
        hist = analyze.extractHSVHist(hsvimg,False)
        if SHOWFLAG:
            analyze.displayHistogram(hist)

        return hist



def scatter(instances,labels):
    if pcaflag:
        analyze.showPCA(instances,labels)

    elif ldaflag:
        optionID = sys.argv.index('lda') + 1
        if optionID <= len(sys.argv) - 1:
            option = sys.argv[optionID]
            analyze.showLDA2(instances,labels,classes=option)
        else:
            analyze.showLDA(instances,labels)


if __name__ == '__main__':

    if len(sys.argv) >= 2:

        #if user input is a single image apply to image
        if os.path.isfile(sys.argv[1]) and os.path.splitext(sys.argv[1])[1] == '.npy':
            tmp = np.load(sys.argv[1])
            instances  = tmp[:,:-1].astype(float)
            labels = tmp[:,-1:].astype(str)

            scatter(instances,labels)

        else:
            #evaluate single image
            #check if the image was read in correctly
            original = cv2.imread(full_path,cv2.IMREAD_COLOR)
            display(sys.argv[1])

    #if less than 3 args given
    else:
        print "wrong number of files as arguments expecting 3:"
        print "argv1 = image file/directory"
        print "argv2 + = modes of operation"
        sys.exit()

    #find out execution time
    print("--- %s seconds ---" % (time.time() - start_time))




