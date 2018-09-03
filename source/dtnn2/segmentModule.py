#############################################################################################################
#Masa Hu
#
#                               Trash Classification Project
#############################################################################################################
#Library imports
import numpy as np
import cv2
import random
import math
import pymeanshift as pms
import os
from matplotlib import pyplot as plt
############################################################################################################
#Flag options for imread are self explanatory
#cv2.IMREAD_GRAYSCALE
#Cv2.IMREAD_COLOR
#cv2.IMREAD_UNCHANGED
#############################################################################################################
#Global Variables
allimages = {}                          #put all images in this dictionary here to show them later
#############################################################################################################
###############################################################################################################################
#Documentation
########################################################################
#BilateralFilter
########################################################################
#http://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?highlight=bilateralfilter#bilateralfilter
########################################################################
#Prameters:
#    src - src image
#    dst - Destination image of the same size and type as src .
#    d - Diameter of each pixel neighborhood that is used during filtering. If it is non-positive, it is computed from sigmaSpace .
#    sigmaColor - Filter sigma in the color space. A larger value of the parameter means that farther colors within the pixel neighborhood (see sigmaSpace ) will be mixed together, resulting in larger areas of semi-equal color.
#    sigmaSpace - Filter sigma in the coordinate space. A larger value of the parameter means that farther pixels will influence each other as long as their colors are close enough (see sigmaColor ). When d>0 , it specifies the neighborhood size regardless of sigmaSpace . Otherwise, d is proportional to sigmaSpace .
#
# bilateralFilter(src, d, sigmaColor, sigmaSpace)

MIN_DENSITY = 10000
SPATIAL_RADIUS = 5
RANGE_RADIUS = 5


#########################################################################################################################
#tile the image
#########################################################################################################################
#tiles the foreground onto the background
def getTiledSegment(segment,fgmask):
    #find a colored spot we can tile the image with
    h,w = segment.shape[:2]

    #FIND A POINT ON THE FOREGROUND. if foreground is not in the center, center the mask and the image
    cy,cx = math.ceil(h/2),math.ceil(w/2)
    if not fgmask[cy,cx]:
        cys,cxs = np.where(fgmask == True)
        idcenter = math.ceil(len(cys) / 2)
        hdiff = abs(cy - cys[idcenter]) * 2
        wdiff = abs(cx - cxs[idcenter]) * 2
        height = h + hdiff
        width = w + wdiff
        centeredmask = np.zeros((height,width))
        centeredsegment = np.zeros((height,width,3))
        if cy < cys[idcenter]:
            hdiff = 0
        if cx < cxs[idcenter]:
            wdiff = 0
        centeredsegment[hdiff:hdiff+h,wdiff:wdiff+w] = segment
        centeredmask[hdiff:hdiff+h,wdiff:wdiff+w] = fgmask

        #reset segment and fgmask and h,w
        segment = centeredsegment
        fgmask = centeredmask.astype(bool)
        h,w = segment.shape[:2]

        cy,cx = math.ceil(h/2),math.ceil(w/2)

    #initialize tiled image/mask and condition for fulfillment
    tiled_img = np.zeros((h*5,w*5,3))
    tiled_mask = np.zeros((h*5,w*5))
    filled = False

    #tile image until no background is left
    while not filled:
        #get pixel location of unfilled background take the points that are within a bounding box
        #TAKE PIXELS WITHIN BOUNDRIES
        ys,xs = np.where(tiled_mask == False)
        tmp_truthmap = np.logical_and(np.logical_and(ys >= 2*h, ys < h*3),np.logical_and(xs >= 2*w,xs < w*3))
        ys = ys[tmp_truthmap]
        xs = xs[tmp_truthmap]
        centerid = int(len(ys) / 2)

        #find bounding boxes for the background, and the portion of the foreground we are using
        xlow = xs[centerid] - math.ceil(w / 2)
        xhigh = xs[centerid] + int(w / 2)
        ylow = ys[centerid] - math.ceil(h / 2)
        yhigh = ys[centerid] + int(h / 2)

        #append the foreground onto the tiled_img and update the tiled_mask
        tmpimg = tiled_img[ylow:yhigh,xlow:xhigh]
        tmpmask = tiled_mask[ylow:yhigh,xlow:xhigh]
        tmpimg[fgmask] = segment[fgmask]
        tmpmask = np.logical_or(tmpmask,fgmask)
        tiled_img[ylow:yhigh,xlow:xhigh] = tmpimg
        tiled_mask[ylow:yhigh,xlow:xhigh] = tmpmask

        #get pixel location of unfilled background take the points that are within a bounding box
        #TAKE PIXELS WITHIN BOUNDRIES
        ys,xs = np.where(tiled_mask == False)
        tmp_truthmap = np.logical_and(np.logical_and(ys >= 2*h, ys < h*3),np.logical_and(xs >= 2*w,xs < w*3))
        ys = ys[tmp_truthmap]
        xs = xs[tmp_truthmap]
        filled = (len(ys) == 0 or len(xs) == 0)

    #put back the foreground ontop of the tiled background
    tmpimg = tiled_img[h*2:h*3,w*2:w*3]
    tmpimg[fgmask] = segment[fgmask]
    tiled_img[h*2:h*3,w*2:w*3] = tmpimg

    #return tiled image as a np.int8
    return tiled_img[h*2:h*3,w*2:w*3].astype(np.uint8)

#saves the tiled foreground images
def saveTiledSegments(img,labels,category='',outdir='testing'):

    uq_labels = np.unique(labels)
    #for each unique marker crop the image with or without background and save it to the output directory
    count = 0
    for l in uq_labels[1:]:
        segment = img.copy()

        #black out the background
        segment[labels != l] = [0,0,0]

        #opencv bounding rect only works on single channel images... we cut out the segment from the image
        blank = img.copy()
        blank = blank - blank
        blank[labels == l] = [255,255,255]
        grey = cv2.cvtColor(blank,cv2.COLOR_BGR2GRAY)
        x,y,w,h = cv2.boundingRect(grey)

        #crop the image and the mask to the segment in question
        cropped = segment[y:y+h,x:x+w]
        mask = labels[y:y+h,x:x+w]

        #tile the foreground onto the background
        tiled = getTiledSegment(cropped,mask == l)

        #save the file with a unique name
        f_out =  str(count) + "_" + category
        fout = os.path.join(outdir,f_out)
        cv2.imwrite(fout,tiled)
        count += 1

#########################################################################################################################
#saves the segments of the original image as png files given the labels
'''
input:  original image
        segment labels with the original image
        output directory
        (option) show flag => default to false
        (option) bg/nbg => default to with True

output: none
console: original number of segments, saved number of segments after size reduction,  image category
'''
def saveSegments(original,labels,out_dir,category,SHOW=False,showbg=False):

    unique_labels = np.unique(labels)
    blank = original - original

    #get the sizes of the discovered segments
    for x in unique_labels:
        #color the blank canvas with the different segments
        b = random.randint(0,255)
        g = random.randint(0,255)
        r = random.randint(0,255)
        blank[labels == x] = [b,g,r]

    #save the blank canvas
    if not os.path.isdir('ms_segmentation'):
        os.makedirs('ms_segmentation')

    fout_original = os.path.join('ms_segmentation',"segmented_" + category)
    cv2.imwrite(fout_original,blank)

    #for each unique marker crop the image with or without background and save it to the output directory
    count = 0
    for l in unique_labels[1:]:
        segment = original.copy()
        segment[labels != l] = [0,0,0]

        #opencv bounding rect only works on single channel images...
        blank = original.copy()
        blank = blank - blank
        blank[labels == l] = [255,255,255]
        grey = cv2.cvtColor(blank,cv2.COLOR_BGR2GRAY)
        x,y,w,h = cv2.boundingRect(grey)

        #crop the image according to the bounding rect. We lose some image quality as we resize the image to 256x256 before we save it
        if(showbg):
            cropped = original[y:y+h,x:x+w]
        else:
            cropped = segment[y:y+h,x:x+w]
        cropped = np.uint8(cropped)

        #save the file with a unique name
        f_out =  str(count) + "_" + category
        fout = os.path.join(out_dir,f_out)
        cv2.imwrite(fout,cropped)
        count += 1

        if(SHOW):
            cv2.imshow(resized)
            cv2.waitKey(0)

    print('segmentation saved')

###############################################################################################################################
########################################################################

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
def getSegments(original, SHOW=False,sr=SPATIAL_RADIUS,rr=RANGE_RADIUS,md=MIN_DENSITY):
    allimages["original"] = original
    ##############################################################################################################
    #gaussian Blur
    #blur = cv2.GaussianBlur(gray_img,(5,5),0)
    #allimages["gaussianBlur"] = blur

    #mean shift segmentation on bgr image
    #https://github.com/fjean/pymeanshift
    #http://ieeexplore.ieee.org/document/1000236/
    segmented_image,labels_image,number_regions = pms.segment(
            original,
            spatial_radius=sr,
            range_radius=rr,
            min_density=md,
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

    allimages["shift segmentation"] = blank
    ################################################################################
    ################################################################################
    ################################################################################
    ################################################################################
    ################################################################################

    return segmented_image, labels_image


#################################################################################
#TEST THE FUNCTIONS
if __name__ == '__main__':

    TESTIMG = 'categories/test_images/lenasmall.jpg'
    DIROUT = 'testing/'
    img = cv2.imread(TESTIMG,cv2.IMREAD_COLOR)
    imgname = "lenasmall.jpg"

    segimg,segmask = getSegments(img,sr=1,rr=1,md=1000)
    saveTiledSegments(img,segmask,category=imgname,outdir='')

    cv2.imshow('original',img)
    cv2.imshow('segmentation', segimg)


