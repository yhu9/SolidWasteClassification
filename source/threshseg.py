import numpy as np
import cv2
import sys
import os
import random
import segmentModule as seg
from matplotlib import pyplot as plt


#user input
nothreshflag = 'nothresh' in sys.argv
hsvsegflag = 'hsvseg' in sys.argv

#CONSTANTS
MIN_DENSITY = 1000
TREEMATTER = [0,0,255]
PLYWOOD = [0,255,0]
CARDBOARD = [255,0,0]
BLACKBAG = [255,255,0]
TRASHBAG = [255,0,255]
BOTTLES = [0,255,255]
CATS=[TREEMATTER,PLYWOOD,CARDBOARD,BLACKBAG,TRASHBAG,BOTTLES]

#majority rule segmentation on cnn output
def majorityseg(original,mask):
    if hsvsegflag:
        hsvimg = cv2.cvtColor(original,cv2.COLOR_BGR2HSV)
        segmented_image, labels = seg.getSegments(hsvimg,md=MIN_DENSITY)
    else:
        segmented_image, labels = seg.getSegments(original,md=MIN_DENSITY)

    unique_labels = np.unique(labels)
    blank1 = original - original
    blank2 = original - original
    for label in unique_labels:

        #randomly paint blank1
        b = random.randint(0,255)
        g = random.randint(0,255)
        r = random.randint(0,255)
        blank1[ labels == label] = [b,g,r]

        #find majority category and paint blank2
        majority = -1
        for cat in CATS:
            tmp = mask[labels == label]
            count = np.count_nonzero(np.all(tmp == cat,axis=1))
            if count > majority:
                classification = cat
                majority = count

        blank2[labels == label] = classification

    cv2.imshow('ms_segmentation',blank1)
    cv2.imshow('majority segmentation',blank2)
    cv2.waitKey(0)

    return blank1,blank2


def threshseg2(original,raw_values,thresh_val=1):

    #segment the image
    if hsvsegflag:
        hsvimg = cv2.cvtColor(original,cv2.COLOR_BGR2HSV)
        segmented_image, labels = seg.getSegments(hsvimg,md=MIN_DENSITY)
    else:
        segmented_image, labels = seg.getSegments(original,md=MIN_DENSITY)

    #get the threshold matrix.
    #1. take sum of all positive predictions divided by the max value. The closer to 1 the better
    h,w,d = raw_values.shape
    raw_values += np.abs(np.min(raw_values,axis=2)).reshape((h,w,1))
    raw_values /= np.max(raw_values,axis=2).reshape((h,w,1))
    raw_values = np.nan_to_num(raw_values)
    #means = np.mean(raw_values,axis=2).reshape((h,w,1))
    #raw_values[raw_values < means] = 0
    thresh_matrix = np.sum(raw_values,axis=2)

    #get the mask
    mask = raw_values.argmax(axis=2)

    #obscure the low threshold pixels
    mask[thresh_matrix > thresh_val] = -1
    mask[thresh_matrix <= 0] = -1

    #display histogram
    h,w = thresh_matrix.shape[:2]
    hist = thresh_matrix.reshape(h * w)
    hist.sort()
    plt.plot(hist)
    plt.show()

    #paint a rgb mask
    rgb_mask = original
    rgb_mask[mask == -1] = [0,0,0]
    rgb_mask[mask == 0] = [0,0,255]
    rgb_mask[mask == 1] = [0,255,0]
    rgb_mask[mask == 2] = [255,0,0]
    rgb_mask[mask == 3] = [0,255,255]
    rgb_mask[mask == 4] = [255,0,255]
    rgb_mask[mask == 5] = [255,255,0]

    #create the segmented image with majority rule using classification categories
    unique_labels = np.unique(labels)
    blank1 = original - original
    blank2 = original - original
    for label in unique_labels:

        #randomly paint blank1 according to meanshift labels
        b = random.randint(0,255)
        g = random.randint(0,255)
        r = random.randint(0,255)
        blank1[labels == label] = [b,g,r]

        #find majority category and paint blank2
        majority = -1
        for cat in CATS:
            tmp = rgb_mask[labels == label]
            count = np.count_nonzero(np.all(tmp == cat,axis=1))
            if count > majority:
                classification = cat
                majority = count
        blank2[labels == label] = classification

    #show the results
    cv2.imshow('ms_segmentation',blank1)
    cv2.imshow('majority segmentation',blank2)
    cv2.imshow('thresholding mask',rgb_mask)
    cv2.waitKey(0)

    return blank1,blank2,rgb_mask


#segmentation with thresholding on raw cnn output
def threshseg(original,raw_values,thresh_val='median'):

    #segment the image
    if hsvsegflag:
        hsvimg = cv2.cvtColor(original,cv2.COLOR_BGR2HSV)
        segmented_image, labels = seg.getSegments(hsvimg,md=MIN_DENSITY)
    else:
        segmented_image, labels = seg.getSegments(original,md=MIN_DENSITY)

    #get the threshold matrix.
    #1. take sum of all positive predictions divided by the max value. The closer to 1 the better
    h,w,d = raw_values.shape
    raw_values += abs(raw_values.min())
    raw_values /= float(raw_values.max())

    #get the mask
    mask = raw_values.argmax(axis=2)

    #get the thresholded matrix
    raw_values.sort(axis=2)
    thresh_matrix = raw_values[:,:,-1] - raw_values[:,:,-2]

    #get threshold value
    medval = np.median(thresh_matrix[30:-30,30:-30])
    minval = thresh_matrix.min()
    maxval = thresh_matrix.max()
    meanval = np.mean(thresh_matrix)
    print('mean: %.4f' % meanval)
    print('min: %.4f' % minval)
    print('med: %.4f' % medval)
    print('max: %.4f' % maxval)
    if(thresh_val == 'median'):
        thresh_val = medval
    elif(thresh_val == 'mean'):
        thresh_val = meanval
    print('threshval: %.4f' % thresh_val)

    #obscure the low threshold pixels
    mask[thresh_matrix < thresh_val] = -1

    #paint a rgb mask
    rgb_mask = original
    rgb_mask[mask == -1] = [0,0,0]
    rgb_mask[mask == 0] = [0,0,255]
    rgb_mask[mask == 1] = [0,255,0]
    rgb_mask[mask == 2] = [255,0,0]
    rgb_mask[mask == 3] = [0,255,255]
    rgb_mask[mask == 4] = [255,0,255]
    rgb_mask[mask == 5] = [255,255,0]

    #create the segmented image with majority rule using classification categories
    unique_labels = np.unique(labels)
    blank1 = original - original
    blank2 = original - original
    for label in unique_labels:

        #randomly paint blank1 according to meanshift labels
        b = random.randint(0,255)
        g = random.randint(0,255)
        r = random.randint(0,255)
        blank1[labels == label] = [b,g,r]

        #find majority category and paint blank2
        majority = -1
        for cat in CATS:
            tmp = rgb_mask[labels == label]
            count = np.count_nonzero(np.all(tmp == cat,axis=1))
            if count > majority:
                classification = cat
                majority = count
        blank2[labels == label] = classification

    #show the results
    cv2.imshow('ms_segmentation',blank1)
    cv2.imshow('majority segmentation',blank2)
    cv2.imshow('thresholding mask',rgb_mask)
    cv2.waitKey(0)

    return blank1,blank2,rgb_mask

######################################################################################################3
######################################################################################################3
######################################################################################################3
######################################################################################################3
if __name__ == '__main__':

    if len(sys.argv) >= 4 and nothreshflag:

        #read the original image and the mask
        if os.path.exists(sys.argv[1]) and os.path.exists(sys.argv[2]):
            original = cv2.imread(sys.argv[1],cv2.IMREAD_COLOR)
            mask = cv2.imread(sys.argv[2],cv2.IMREAD_COLOR)
        else:
            print('ERROR READING THE FILES: %s, %s' % (sys.argv[1],sys.argv[2]))
            quit()

        #make sure the shapes are the same
        h1,w1 = original.shape[:2]
        h2,w2 = mask.shape[:2]
        if h1 != h2 or w1 != w2:
            img= cv2.resize(original,(h2,w2),interpolation=cv2.INTER_CUBIC)
        else:
            img = original

        #get the majority rule segmentation
        ms_segmentation, majority_segmentation = majorityseg(img,mask)

        #check for the existence of the results directory
        if not os.path.exists('results'):
            os.makedirs('results')

        #write resulting images in the results results directory
        fout1 = os.path.join('results','meanshift_' + os.path.splitext(os.path.basename(sys.argv[1]))[0] + '.png')
        fout2 = os.path.join('results','majoritysegmentation_' + os.path.splitext(os.path.basename(sys.argv[1]))[0] + '.png')
        cv2.imwrite(fout1,ms_segmentation)
        cv2.imwrite(fout2,majority_segmentation)

    elif len(sys.argv) >= 3 and os.path.exists(sys.argv[1]) and os.path.exists(sys.argv[2]) and os.path.splitext(os.path.basename(sys.argv[2]))[1] == '.npy':
        #read in the image and the raw values
        original = cv2.imread(sys.argv[1],cv2.IMREAD_COLOR)
        raws = np.load(sys.argv[2]).astype(np.float32)

        #make sure the images are the same size
        h1,w1 = original.shape[:2]
        h2,w2 = raws.shape[:2]
        if h1 != h2 or w1 != w2:
            img= cv2.resize(original,(h2,w2),interpolation=cv2.INTER_CUBIC)
        else:
            img = original

        #read the threshold option if it exists
        if 'thresh' in sys.argv and len(sys.argv) > 4:
            index = sys.argv.index('thresh')
            index += 1
            thresh_val = sys.argv[index]

            #get the thresholding segmentation
            if len(sys.argv) > 4 and sys.argv[3] == 'masathresh':
                ms_segmentation, majority_segmentation,thresh_mask = threshseg2(img,raws,thresh_val=float(thresh_val))
            else:
                ms_segmentation, majority_segmentation,thresh_mask = threshseg(img,raws,thresh_val=float(thresh_val))
        else:
            if len(sys.argv) == 4 and sys.argv[3] == 'masathresh':
                ms_segmentation, majority_segmentation,thresh_mask = threshseg2(img,raws)
            else:
                ms_segmentation, majority_segmentation,thresh_mask = threshseg(img,raws)

        #create the results directory
        if not os.path.exists('results'):
            os.makedirs('results')

        #write resulting images in the results results directory
        fout1 = os.path.join('results','meanshift_' + os.path.splitext(os.path.basename(sys.argv[1]))[0] + '.png')
        fout2 = os.path.join('results','threshmajority' + os.path.splitext(os.path.basename(sys.argv[1]))[0] + '.png')
        fout3 = os.path.join('results','threshmask_' + os.path.splitext(os.path.basename(sys.argv[1]))[0] + '.png')
        cv2.imwrite(fout1,ms_segmentation)
        cv2.imwrite(fout2,majority_segmentation)
        cv2.imwrite(fout3,thresh_mask)
    else:
        print('error with python arguments to program')
        print('expecting:')
        print('python threshseg.py [img_dir] [rawvalues_dir] [thresh/nothresh/masathresh]')




