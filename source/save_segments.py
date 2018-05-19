import segmentModule
import sys
import os
import cv2

OUT_DIR = sys.argv[3]

#for each image in directory get the segments and save them
'''
inputs:
    1. input directory with all images to process
    2. (option) with or without background of each segment
output:
    none
'''
def saveMSSegments(directory,bg=False):

    cat1_list = os.listdir(directory)
    for f1 in cat1_list:
        full_dir1 = directory + f1

        image1 = cv2.imread(full_dir1,cv2.IMREAD_COLOR)

        original, labels = segmentModule.getSegments(image1,False,rr=1,sr=1)
        if(bg):
            segmentModule.saveSegments(original,labels,OUT_DIR,f1)
        else:
            segmentModule.saveSegments(original,labels,OUT_DIR,f1,showbg=False)

#for each image in drectory get the HSV segments and save them
'''
inputs:
    1. input directory with all images to process
    2. (option) with or without background of each segment
output:
    none
'''
def saveMSSegmentsHSV(directory,bg=False):
    cat1_list = os.listdir(directory)
    for f1 in cat1_list:
        full_dir1 = directory + f1

        image1 = cv2.imread(full_dir1,cv2.IMREAD_COLOR)
        img = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)

        original, labels = segmentModule.getSegments(img,False,rr=5,sr=5)
        if(bg):
            segmentModule.saveSegments(original,labels,OUT_DIR,f1)
        else:
            segmentModule.saveSegments(original,labels,OUT_DIR,f1,showbg=False)

def saveMSSegmentsBGRHSV(directory,bg=False):
    cat1_list = os.listdir(directory)
    for f1 in cat1_list:
        full_dir1 = directory + f1

        image1 = cv2.imread(full_dir1,cv2.IMREAD_COLOR)
        img = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)

        original, labels = segmentModule.getSegments(img,False,rr=5,sr=5)
        if(bg):
            segmentModule.saveSegments(image1,labels,OUT_DIR,f1)
        else:
            segmentModule.saveSegments(image1,labels,OUT_DIR,f1,showbg=False)

#helper function which flips an image
'''
input:
    1. cv image
output:
    1. image flipped horizontally
    2. image flipped vertically
    3. image flipped diagnolly
'''
def flipImage(img):
    horizontal = cv2.flip(img,0)
    vertical = cv2.flip(img,1)
    diagnol = cv2.flip(img,-1)

    return horizontal, vertical, diagnol

#helper function which rotates an image
'''
input:
    1. cv image
output:
    1. image rotated 90 degrees
    2. image rotated 180 degrees
    3. image rotated 270 degrees
'''
def rotateImage(img):
    rows,cols,depth = img.shape
    M1 = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
    M2 = cv2.getRotationMatrix2D((cols/2,rows/2),180,1)
    M3 = cv2.getRotationMatrix2D((cols/2,rows/2),270,1)
    dst1 = cv2.warpAffine(original,M1,(cols,rows))
    dst2 = cv2.warpAffine(original,M1,(cols,rows))
    dst3 = cv2.warpAffine(original,M1,(cols,rows))

    return dst1, dst2, dst3

#for each image in directory rotate the image in four different ways and mirror it in 3 different ways and save them in the same directory
'''
input:
    1. image directory with images to rotate/mirror
output:
    none
'''
def multiplyImages(image_dir):
    args = os.listdir(image_dir)
    for f in args:
        full_dir = image_dir + f

        original = cv2.imread(full_dir,cv2.IMREAD_COLOR)

        f1,f2,f3 = flipImage(original)
        for img,s in zip([f1,f2,f3,original],['h','v','b','o']):
            rows,cols,depth = img.shape

            r1,r2,r3 = rotateImage(img)

            #cv2.imshow('warped',dst)
            dst_dir1 = image_dir + str(90) + "_" + s + "_" + f
            dst_dir2 = image_dir + str(180) + "_" + s + "_" + f
            dst_dir3 = image_dir + str(270) + "_" + s + "_" + f

            cv2.imwrite(dst_dir1,r1)
            cv2.imwrite(dst_dir2,r2)
            cv2.imwrite(dst_dir3,r3)

        dst_dir1 = image_dir + "h_" + f
        dst_dir2 = image_dir + "v_" + f
        dst_dir3 = image_dir + "b_" + f
        cv2.imwrite(dst_dir1,f1)
        cv2.imwrite(dst_dir2,f2)
        cv2.imwrite(dst_dir3,f3)


#####################################################################################################################################33
#####################################################################################################################################33
#####################################################################################################################################33
#####################################################################################################################################33
#####################################################################################################################################33

#main function

#save rgb
showbg = 'showbg' in sys.argv
if len(sys.argv) >= 3 and (sys.argv[1] == 'save' or sys.argv[1] == 'savergb'):
    if(len(sys.argv) > 3):
        saveMSSegments(sys.argv[2],bg=showbg)
    else:
        saveMSSegments(sys.argv[2],bg=showbg)

#save hsv
elif len(sys.argv) >= 3 and sys.argv[1] == 'savehsv':
    if(len(sys.argv) > 3):
        saveMSSegmentsHSV(sys.argv[2],bg=showbg)
    else:
        saveMSSegmentsHSV(sys.argv[2],bg=showbg)

#extract hsv blobs , but save bgr blobs
elif len(sys.argv) >= 3 and sys.argv[1] == 'bgrhsv':
    saveMSSegmentsBGRHSV(sys.argv[2],bg=showbg)

#save rotations
elif len(sys.argv) == 3 and sys.argv[1] == 'rotate':
    multiplyImages(sys.argv[2])

#otherwise the mode is wrong
else:
    print("WRONG MODE and args")
    print("arg1 = 'save','rotate'")
    print("arg2 = 'file directory'")

