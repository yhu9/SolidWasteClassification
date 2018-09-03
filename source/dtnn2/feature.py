import numpy as np
import cv2
import os
import time
import random
import segmentModule as seg
import constants
import signed_difference as sd
from sklearn import preprocessing

#creates the testing instances from the image by extracting the blobs and evaluating hog/color/gabor features
def extractImage(img,imgname,n = 'all'):

    #read in a mixed image and its ground truth
    gt = np.zeros(img.shape)

    #segment the image and extract blobs
    blobs,labels,markers = extractTestingBlobs(img,gt)

    #evaluate each blob and extract features from them as well as its label by looking at the ground truth
    instances = []
    i = 0
    for blob in blobs:
        i+=1
        featurevector = getFeatureVector(blob)
        instances.append(featurevector)
        print("extracting Features: %i of %i segments " % (i,len(blobs)))

    #return the training instances and the labels
    if n == 'all':
        return np.array(instances), np.array(labels),markers
    else:
        return np.array(instances)[:n],np.array(labels)[:n]

#creates the testing instances from the image by extracting the blobs and evaluating hog/color/gabor features
def getTestingBatch(imgname=constants.MIXED_FILE,gtname=constants.GROUND_TRUTH,n = 'all'):

    #read in a mixed image and its ground truth
    tmp1 = cv2.imread(imgname,cv2.IMREAD_COLOR)
    tmp2 = cv2.imread(gtname,cv2.IMREAD_COLOR)
    image = cv2.resize(tmp1,(constants.FULL_IMGSIZE,constants.FULL_IMGSIZE),interpolation=cv2.INTER_CUBIC)
    gt = cv2.resize(tmp2,(constants.FULL_IMGSIZE,constants.FULL_IMGSIZE),interpolation=cv2.INTER_NEAREST)
    gt[gt <= 128] = 0
    gt[gt > 128] = 255

    #segment the image and extract blobs
    blobs,labels,markers = extractTestingBlobs(image,gt)

    #evaluate each blob and extract features from them as well as its label by looking at the ground truth
    instances = []
    i = 0
    for blob in blobs:
        i+=1
        featurevector = getFeatureVector(blob)
        instances.append(featurevector)
        print("extracting Features: %i of %i segments " % (i,len(blobs)))

    #return the training instances and the labels
    if n == 'all':
        return np.array(instances), np.array(labels)
    else:
        return np.array(instances)[:n],np.array(labels)[:n]


#helper function for finding the label of a segmentation based on the file name
def getCatFromName(filename):
    if filename.find('treematter') > 0:
        return 0
    elif filename.find('plywood') > 0:
        return 1
    elif filename.find('cardboard') > 0:
        return 2
    elif filename.find('bottles') > 0:
        return 3
    elif filename.find('trashbag') > 0:
        return 4
    elif filename.find('blackbag') > 0:
        return 5
    else:
        return -1

#creates the training instances from the different categories
def getTrainingBatch(n):
    #get the category directories
    segmentnames = os.listdir(constants.segment_dir)
    instances = []
    labels = []
    if n > len(segmentnames):
        n = len(segmentnames)

    #go through each file and extract features from them
    #something like 200000 segments
    i = 0
    random.shuffle(segmentnames)
    for f in segmentnames:
        i += 1
        full_dir = os.path.join(constants.segment_dir,f)
        seg = cv2.imread(full_dir,cv2.IMREAD_COLOR)
        cat = constants.CATS_ONEHOT[getCatFromName(f)]
        featurevector = getFeatureVector(seg)
        instances.append(featurevector)
        labels.append(cat)
        #print("%i of %i training segments DONE" % (i,n))
        if i >= n: break

    #return the instances and their labels as numpy arrays
    return np.array(instances), np.array(labels)

#extracts blobs from the mixed evaluation image and its ground truth label for testing the model
'''
INPUTS:
    1. image
OUTPUTS:
    1. 3d numpy array with blobs as instances as 2d numpy arrays
    2. 1d numpy array as labels
'''
def extractTestingBlobs(img,gt,hsv=False):
    instances = []
    labels = []
    if hsv:
        print('HSV SEGMENTATION')
        tmp = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        seg_img,markers = seg.getSegments(tmp,False,sr=5,rr=5,md=1000)
    else:
        print('BGR SEGMENTATION')
        seg_img,markers = seg.getSegments(img,False,sr=1,rr=1,md=1000)

    #go through each segment
    marks = np.unique(markers)
    for uq_mark in marks:
        #get the segment and append it to inputs
        region = img.copy()
        region[markers != uq_mark] = [0,0,0]
        gtregion = gt.copy()
        gtregion[markers != uq_mark] = [0,0,0]

        #opencv bounding rect only works on single channel images...
        blank = img.copy()
        blank = blank - blank
        blank[markers == uq_mark] = [255,255,255]
        grey = cv2.cvtColor(blank,cv2.COLOR_BGR2GRAY)
        x,y,w,h = cv2.boundingRect(grey)

        #crop the colored region
        cropped = region[y:y+h,x:x+w]
        mask = markers[y:y+h,x:x+w]

        #tile the cropped region
        segment = seg.getTiledSegment(cropped,mask == uq_mark)

        #find out what the label is
        majority = -1
        for i,cat in enumerate(constants.CATS):
            tmp = gtregion[markers == uq_mark]
            count = np.count_nonzero(np.all(tmp == cat,axis=1))
            if count > majority:
                classification = constants.CATS_ONEHOT[i]
                majority = count

        #append the instance and its classification label
        labels.append(classification)
        instances.append(segment.astype(np.uint8))

    #return the testing instances and labels
    return instances, labels, markers

# Get patch returns a cropped portion of the image provided using the globally defined radius
# pixel is a tuple of (row, column) which is the row number and column number of the pixel in the picture
# image is a cv2 image
def get_patch(pixel, image, height, width, sd_matrix):
	radius = 6  # Used for patch size
	diameter = 2 * radius
	# max_row, max_col, not_used = np.array(image).shape Having this was making it super slow, so just manually put
	# in the size of the images i guess
	max_row = height
	max_col = width
	if pixel[0] >= (max_row - radius):
		corner_row = max_row - (diameter + 2)
	elif pixel[0] >= radius:
		corner_row = pixel[0] - radius
	else:  # With the row coordinate being less than the radius of the patch, it has to be at the top of the image
		corner_row = 0  # meaning the row coordinate for the patch will have to be 0

	if pixel[1] >= (max_col - radius):
		corner_col = max_col - (diameter + 2)
	elif pixel[1] >= radius:
		corner_col = pixel[1] - radius
	else:  # With the column coordinate being less than the radius of the patch, it has to be in the left side of the
		corner_col = 0  # Image, meaning the column coordinate for the patch will have to be 0
	diameter += 1  # Added 1 for the center pixel

	return image[corner_row:(corner_row + diameter), corner_col:(corner_col + diameter)], sd_matrix[corner_row:(
			corner_row + diameter), corner_col:(corner_col + diameter)]


def k_means_color(patch):
	z = patch.reshape((-1, 3))
	# Set the rgb values to be floats so it can be used in the k-means function
	z = np.float32(z)
	# Create the criteria for k-means clustering, 1st: Stop kmeans when the specified accuracy is met, or when the
	# max iterations specified is met. 2nd: max iterations. 3rd: epsilon, or required accuracy
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	k = 2
	# run the K means clustering using cv2, so it can be done easily with images
	# label and center being the important returns, with label being important for producing an image to show the clusters
	# and center being useful for the NN and the producing the image to show the clusters, as its the average color of each
	# cluster. Arguments for the kmeans, 1st: input data, 2nd: number of clusters needed, 3rd: not sure,
	# 4th: the criteria specified above, 5th: number of times to run the clustering taking the best result, 6th: flags
	ret, label, center = cv2.kmeans(z, k, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)
	center = np.uint8(center)  # Center will contain the Dominant Colors in their respective color channels
	# i.e [[DCb, DCg, DCr], [DCb, DCg, DC3r]] with k = 2
	return center.flatten()

# Returns the dominate colors in a patch, which are the average colors based upon what is the center of clusters that
# are built from the rgb values in the patch, patch is a 3d array which is 50x50x3
def get_dominate_color(patch):
	b_root = patch[:, :, 0]
	g_root = patch[:, :, 1]
	r_root = patch[:, :, 2]

	b_root_mean = np.mean(b_root)
	g_root_mean = np.mean(g_root)
	r_root_mean = np.mean(r_root)

	b_child_0 = b_root[b_root > b_root_mean]
	b_child_1 = b_root[b_root <= b_root_mean]

	if b_child_0.size == 0:
		half = b_root.size // 2
		b_child_0 = b_root[:half]
		b_child_1 = b_root[half:]

	g_child_0 = g_root[g_root > g_root_mean]
	g_child_1 = g_root[g_root <= g_root_mean]

	if g_child_0.size == 0:
		half = g_root.size // 2
		b_child_0 = g_root[:half]
		b_child_1 = g_root[half:]

	r_child_0 = r_root[r_root > r_root_mean]
	r_child_1 = r_root[r_root <= r_root_mean]

	if r_child_0.size == 0:
		half = r_root.size // 2
		b_child_0 = r_root[:half]
		b_child_1 = r_root[half:]

	center = [np.mean(b_child_0), np.mean(g_child_0), np.mean(r_child_0), np.mean(b_child_1), np.mean(g_child_1),
	          np.mean(r_child_1)]

	return center

def get_texture(sd_patch):
	blue = sd_patch[:, :, 0]
	green = sd_patch[:, :, 1]
	red = sd_patch[:, :, 2]

	r_values, r_counts = np.unique(red, return_counts=True)
	b_values, b_counts = np.unique(blue, return_counts=True)
	g_values, g_counts = np.unique(green, return_counts=True)


	r_neg_len = len(r_values[r_values < 0])
	b_neg_len = len(b_values[b_values < 0])
	g_neg_len = len(g_values[g_values < 0])

	r_neg_count = r_counts[:r_neg_len]
	r_pos_count = r_counts[r_neg_len:]

	r_neg_divisor = np.sum(r_neg_count)
	r_pos_divisor = np.sum(r_pos_count)

	b_neg_count = b_counts[:b_neg_len]
	b_pos_count = b_counts[b_neg_len:]

	b_neg_divisor = np.sum(b_neg_count)
	b_pos_divisor = np.sum(b_pos_count)

	g_neg_count = g_counts[:g_neg_len]
	g_pos_count = g_counts[g_neg_len:]

	g_neg_divisor = np.sum(g_neg_count)
	g_pos_divisor = np.sum(g_pos_count)

	r_neg_prob = r_neg_count / r_neg_divisor
	r_pos_prob = r_pos_count / r_pos_divisor

	b_neg_prob = b_neg_count / b_neg_divisor
	b_pos_prob = b_pos_count / b_pos_divisor

	g_neg_prob = g_neg_count / g_neg_divisor
	g_pos_prob = g_pos_count / g_pos_divisor
	return np.array([np.sum(r_neg_prob**2), np.sum(r_pos_prob**2), np.sum(b_neg_prob**2), np.sum(b_pos_prob**2),
	                 np.sum(g_neg_prob**2), np.sum(g_pos_prob**2)])

def run_pixels(image, data, sd_matrix):
	h, w = image.shape[:2]  # getting the height and width of the image for the patch calculations
	return_array = []
	texture = []
	color = []
	coordinates = data[:, 1:]  # removing the label for the data
	for coordinate in coordinates:
		patch, sd_patch = get_patch(coordinate, image, h, w, sd_matrix)
		descriptor_color = k_means_color(patch)
		descriptor_texture = get_texture(sd_patch)
		texture.append(descriptor_texture)
		color.append(descriptor_color)
	return_array.extend(np.concatenate((texture, color), axis=1))
	return np.array(return_array)

def run_image(image, sd_matrix):
	h, w = image.shape[:2]  # getting the height and width of the image for the patch calculations
	return_array = []
	texture = []
	color = []
	for i in range(h):
		for j in range(w):
			coordinate = (i, j)
			patch, sd_patch = get_patch(coordinate, image, h, w, sd_matrix)
			descriptor_color = k_means_color(patch)
			descriptor_texture = get_texture(sd_patch)
			texture.append(descriptor_texture)
			color.append(descriptor_color)
	return_array.extend(np.concatenate((texture, color), axis=1))
	return np.array(return_array)

#uses the feature extractor to extract features from an image
def getFeatureVector(blob):
    #get texture only works on patches
    sdmatrix = sd.getSDMatrix(blob)
    vec1 = get_texture(sdmatrix)

    #get color
    vec2 = k_means_color(blob)

    #return featurevector of a blob
    return np.concatenate((np.array(vec1),np.array(vec2)))

#unit testing
if __name__ == '__main__':

    #test get texture
    img = cv2.imread('../categories/test_images/lenasmall.jpg',cv2.IMREAD_COLOR)
    sdmatrix = sd.getSDMatrix(img)
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('test 1 get_texture(sd_img): ')
    print('---------------------------')
    print(get_texture(sdmatrix))
    print('---------------------------')
    print('---------------------------')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')


