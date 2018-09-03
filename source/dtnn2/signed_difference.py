import numpy as np
import cv2
import os
import sys
import math



#get the 8 radial offsets for calculating the image adjustments
#input:
#   integer
#
#output:
#   numpy array of shape (8,2)
#
def getRadialOffsets(radius):
    p = 8
    k = np.arange(p) + 1
    a_k = [((k-1) * 2 * np.pi) / p]
    x = radius * np.cos(a_k)
    y = -radius * np.sin(a_k)

    return np.concatenate((x.T,y.T),axis=1)

#masa's implementation for acquiring the signed difference matrix of an image
#get_matrix creates a signed difference matrix ofan image given as input
#input:
#   np.array with 3-dimensions typical of an image (h,w,channels)
#
#outputs:
#   np.array with 3-dimensions (h,w,channels,bincount=8)
#
def getSDMatrix(img):

    #some error checking for the radius of the patches
    h,w = img.shape[:2]
    if h <= 4 or w <=4:
        r = min(h,w) - 1
    else:
        r = 4

    #get the offsets from each pixel
    radialoffsets = getRadialOffsets(r)

    #initialize signed difference arrays
    sds = np.empty((h,w,3,len(radialoffsets)))

    #get the sd array for each point before we combine them all
    for i,offset in enumerate(radialoffsets):
        matrix = np.zeros((h + int(r * 2),w+int(r*2),3))
        xoffset = offset[0]
        yoffset = offset[1]

        #adjust the image according to the offset of the radial points
        tmph,tmpw = matrix.shape[:2]
        cx1 = w/2
        cx2 = tmpw/2
        cy1 = h/2
        cy2 = tmph/2
        cylow = cy2-cy1 + yoffset
        cxlow = cx2-cx1 + xoffset
        matrix[int(cylow):int(cylow) + h,int(cxlow):int(cxlow)+w] = img

        #for debugging or for fun
        #cv2.imshow('matrix',matrix.astype(np.uint8))
        #cv2.waitKey(0)

        #grab the signed differences of each color channel
        offsetimg = matrix[int(cy2-cy1):math.ceil(cy2+cy1),int(cx2-cx1):math.ceil(cx2+cx1)]
        rdiff = offsetimg[:,:,2] - img[:,:,2]
        gdiff = offsetimg[:,:,1] - img[:,:,1]
        bdiff = offsetimg[:,:,0] - img[:,:,0]

        #set the sds matrix to the signed differences for each color channel
        sds[:,:,2,i] = rdiff
        sds[:,:,1,i] = gdiff
        sds[:,:,0,i] = bdiff

    return sds

#a little test on the signed difference function on an image of lena adjust file directory of image accordingly to use this test
if __name__ == '__main__':

    img = cv2.imread('../categories/test_images/lenasmall.jpg',cv2.IMREAD_COLOR)

    matrix=getSDMatrix(img)

    #for our viewing pleasure, and verification of the solution
    for i in range(8):
        cv2.imshow('signed difference image blue',matrix[:,:,0,i].astype(np.uint8))
        cv2.imshow('signed difference image red',matrix[:,:,1,i].astype(np.uint8))
        cv2.imshow('signed difference image green',matrix[:,:,2,i].astype(np.uint8))
        cv2.waitKey(0)

