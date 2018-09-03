import segmentModule
import sys
import os
import cv2

#for each image in directory get the segments and save them
'''
inputs:
    1. input directory with all images to process
    2. (option) with or without background of each segment
output:
    none
'''
def saveMSSegments(directory,dirout='',bg=False):

    cat1_list = os.listdir(directory)
    for f1 in cat1_list:
        full_dir1 = directory + f1
        fileout = os.path.splitext(f1)[0] + '.png'

        tmp = cv2.imread(full_dir1,cv2.IMREAD_COLOR)
        original = cv2.resize(tmp,(1000,1000),interpolation=cv2.INTER_CUBIC)

        segmented_image, labels = segmentModule.getSegments(original,False,md=1000,rr=1,sr=1)
        segmentModule.saveSegments(original,labels,dirout,fileout,showbg=bg)

#saves the tiled segmentations of the images
def saveTiledSegments(directory,dirout='',bg=False):

    cat1_list = os.listdir(directory)
    for f1 in cat1_list:
        full_dir1 = directory + f1
        img = cv2.imread(full_dir1,cv2.IMREAD_COLOR)

        original = cv2.resize(img,(1000,1000),interpolation=cv2.INTER_CUBIC)

        segimg,segmask = segmentModule.getSegments(original,sr=1,rr=1,md=1000)
        segmentModule.saveTiledSegments(original,segmask,category=f1,outdir=dirout)

#for each image in drectory get the HSV segments and save them
'''
inputs:
    1. input directory with all images to process
    2. (option) with or without background of each segment
output:
    none
'''
def saveMSSegmentsHSV(directory,dirout='',bg=False):
    cat1_list = os.listdir(directory)
    for f1 in cat1_list:
        full_dir1 = directory + f1
        fileout = os.path.splitext(f1)[0] + '.png'

        image1 = cv2.imread(full_dir1,cv2.IMREAD_COLOR)
        original = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)

        seg_img, labels = segmentModule.getSegments(original,False,rr=5,sr=5)
        segmentModule.saveSegments(original,labels,dirout,fileout,showbg=bg)

def saveMSSegmentsBGRHSV(directory,dirout='',bg=False):
    cat1_list = os.listdir(directory)
    for f1 in cat1_list:
        full_dir1 = directory + f1
        fileout = os.path.splitext(f1)[0] + '.png'

        bgr_img = cv2.imread(full_dir1,cv2.IMREAD_COLOR)
        bgr_small = cv2.resize(bgr_img,(1000,1000),interpolation=cv2.INTER_CUBIC)
        hsv_img = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2HSV)

        seg_img, labels = segmentModule.getSegments(hsv_img,False,md=1000,rr=5,sr=5)
        segmentModule.saveSegments(bgr_small,labels,dirout,fileout,showbg=bg)

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

    return horizontal

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
    dst1 = cv2.warpAffine(img,M1,(cols,rows))
    dst2 = cv2.warpAffine(img,M1,(cols,rows))
    dst3 = cv2.warpAffine(img,M1,(cols,rows))

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
        full_dir = os.path.join(image_dir, f)

        original = cv2.imread(full_dir,cv2.IMREAD_COLOR)

        flipped = flipImage(original)
        r1,r2,r3 = rotateImage(original)
        r4,r5,r6 = rotateImage(flipped)

        #cv2.imshow('warped',dst)
        dst_dir1 = os.path.join(image_dir, str(90) + "_o_" + f)
        dst_dir2 = os.path.join(image_dir, str(180) + "_o_" + f)
        dst_dir3 = os.path.join(image_dir, str(270) + "_o_" + f)
        dst_dir4 = os.path.join(image_dir, str(90) + "_m_" + f)
        dst_dir5 = os.path.join(image_dir, str(180) + "_m_" + f)
        dst_dir6 = os.path.join(image_dir, str(270) + "_m_" + f)
        dst_dir = os.path.join(image_dir, "m_" + f)

        cv2.imwrite(dst_dir1,r1)
        cv2.imwrite(dst_dir2,r2)
        cv2.imwrite(dst_dir3,r3)
        cv2.imwrite(dst_dir4,r4)
        cv2.imwrite(dst_dir5,r5)
        cv2.imwrite(dst_dir6,r6)
        cv2.imwrite(dst_dir,flipped)

#####################################################################################################################################33
#####################################################################################################################################33
##############################6######################################################################################################33
#####################################################################################################################################33
#####################################################################################################################################33

#main function

#save rgb
showbg = 'showbg' in sys.argv

if __name__ == '__main__':
    if len(sys.argv) >= 4:
        if not os.path.isdir(sys.argv[2]):
            print('error opening image directory: %s' % sys.argv[2])
            quit()

        if not os.path.isdir(sys.argv[3]):
            os.makedirs(sys.argv[3])

    if len(sys.argv) >= 4 and (sys.argv[1] == 'save' or sys.argv[1] == 'savergb'):
        saveMSSegments(sys.argv[2],dirout=sys.argv[3],bg=showbg)

    elif len(sys.argv) >= 4 and (sys.argv[1] == 'savetile'):
        saveTiledSegments(sys.argv[2],dirout=sys.argv[3],bg=showbg)

    #save hsv
    elif len(sys.argv) >= 4 and sys.argv[1] == 'savehsv':
        saveMSSegmentsHSV(sys.argv[2],dirout=sys.argv[3],bg=showbg)

    #extract hsv blobs , but save bgr blobs
    elif len(sys.argv) >= 4 and sys.argv[1] == 'bgrhsv':
        saveMSSegmentsBGRHSV(sys.argv[2],dirout=sys.argv[3],bg=showbg)

    #save rotations
    elif len(sys.argv) == 3 and sys.argv[1] == 'rotate':
        multiplyImages(sys.argv[2])

    elif sys.argv[1] =='debug':
        print(sys.argv[3])

    #otherwise the mode is wrong
    else:
        print("WRONG MODE and args")
        print("arg1 = 'save','rotate'")
        print("arg2 = 'file directory'")

