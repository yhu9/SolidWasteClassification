import sys
import os
import cv2


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

        flipped = cv2.flip(original,0)
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


if __name__ == '__main__':

    #save rotations
    if len(sys.argv) == 3 and sys.argv[1] == 'rotate':
        multiplyImages(sys.argv[2])
    else:
        print('end')
