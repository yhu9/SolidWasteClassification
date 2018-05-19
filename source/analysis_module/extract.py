#############################################################################################################
#Masa Hu
#
#                               Trash Classification Project
#############################################################################################################
#Library imports
import sys
import segmentModule
import extractionModule as analyze
import numpy as np
import cv2
import time
import gabor_threads_roi as gabor

#############################################################################################################
#Check system argument length and mode
#if mode is bin do 3d color binning
start_time = time.time()
if(len(sys.argv) >= 3 and (sys.argv[1] == 'b' or sys.argv[1] == 'r' or sys.argv[1] == 'g' or sys.argv[1] == 'bin')):
    mode = str(sys.argv[1])
    imageFileIn = str(sys.argv[2])

    if len(imageFileIn) == 0:
        print("image file name or file name out is empty")
        sys.exit()

    #original = cv2.imread(imageFileIn,cv2.IMREAD_GRAYSCALE)
    original = cv2.imread(imageFileIn,cv2.IMREAD_COLOR)
    #IF i can figure out how to do a 3-d overlay ontop of a 2-d image
    #rgbimage = cv2.cvtColor(original,cv2.COLOR_BGR2RGB)
    analyze.showImage3D(original,mode)

#if mode is hog, show hog feature vector of image
elif len(sys.argv) >= 3 and sys.argv[1] == 'hog':
    mode = sys.argv[1]
    imageFileIn = str(sys.argv[2])

    if len(imageFileIn) == 0:
        print("image file name of file name out is empty")
        sys.exit()

    original = cv2.imread(imageFileIn,cv2.IMREAD_COLOR)
    analyze.extractHOG(original,True)

#if mode is color, show color histogram of image
elif len(sys.argv) >= 3 and sys.argv[1] == 'color':
    mode = sys.argv[1]
    imageFileIn = str(sys.argv[2])
    if len(imageFileIn) == 0:
        print("name of image file is empth")
        sys.exit()

    original = cv2.imread(imageFileIn,cv2.IMREAD_COLOR)
    analyze.extractColorHist(original,True)

#if mode is meanshift, apply meanshift
elif len(sys.argv) == 3 and sys.argv[1] == 'meanshift':
    imageFileIn = str(sys.argv[2])
    original = cv2.imread(imageFileIn,cv2.IMREAD_COLOR)
    image, labels = analyze.meanshift(original)
    segmentModule.showSegments(image,labels)
    print(labels)

#if mode is meanshiftbin, convert 2d image to 3d using bin method and apply meanshift
elif len(sys.argv) == 3 and sys.argv[1] == 'meanshiftbin':
    imageFileIn = str(sys.argv[2])
    original = cv2.imread(imageFileIn,cv2.IMREAD_COLOR)
    image,labels = analyze.meanshift(original,binning=True)
    segmentModule.showSegments(image,labels)
    print(labels)

#if mode is fjmeanshift, do fjmeanshift
elif len(sys.argv) == 3 and sys.argv[1] == 'fjmeanshift':
    imageFileIn = str(sys.argv[2])
    original = cv2.imread(imageFileIn,cv2.IMREAD_COLOR)
    image,labels = segmentModule.getSegments(original,True)

#if mode is meanshiftbin, convert 2d image to 3d using bin method and apply meanshift
elif len(sys.argv) == 3 and sys.argv[1] == 'dbscan':
    imageFileIn = str(sys.argv[2])
    original = cv2.imread(imageFileIn,cv2.IMREAD_COLOR)
    image,labels = analyze.dbscan(original,binning=True)
    segmentModule.showSegments(image,labels)
    print(labels)

#if mode is gabor, extract gabor feature from image using several orientations
elif len(sys.argv) == 3 and sys.argv[1] == 'gabor':
    imageFileIn = sys.argv[2]
    combined_filename = sys.argv[2]
    image_color = cv2.imread(imageFileIn,cv2.IMREAD_COLOR)
    orientations = 16
    filters = gabor.build_filters(orientations)
    mask =
    if image_color == None:
        print('invalid image! Could not open: %s' % imageFileIn)
        sys.exit()

    # Generate and save ALL hogs for this image
    result = gabor.run_gabor(image_color, filters, mask, combined_filename, orientations, mode='training')

else:
    print "wrong number of files as arguments expecting 3:"
    print "argv1 = imageFileIn"
    sys.exit()

#find out execution time
print("--- %s seconds ---" % (time.time() - start_time))

